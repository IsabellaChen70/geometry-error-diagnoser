#!/usr/bin/env python3
"""Offline regression tests for the resumable LLM judge."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_llm_judge as judge


HERE = Path(__file__).resolve().parent
RAW_EVIDENCE_PATH = HERE / "LLM_JUDGE_RAW.jsonl"


def valid_judgment(case_id: object = 129) -> dict:
    value = json.loads(json.dumps(judge.strict_output_template(case_id)))
    value["tutoring_preference"] = "tie"
    value["overall_preference"] = "tie"
    return value


class JudgmentParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_rows = judge.load_jsonl(RAW_EVIDENCE_PATH)
        cls.live_row = next(
            row
            for row in cls.raw_rows
            if row.get("provider") == "tfy"
            and row.get("model") == "gpt-5.6-sol"
        )

    def test_all_three_persisted_gpt_5_6_sol_responses_parse(self) -> None:
        attempts = self.live_row["attempts"]
        self.assertEqual(len(attempts), 3)
        for attempt in attempts:
            with self.subTest(attempt=attempt["attempt"]):
                parsed = judge.parse_judgment(
                    attempt["response_text"], self.live_row["case_id"]
                )
                self.assertEqual(
                    parsed["candidate_A"]["leak_types"],
                    ["exact_reflection_line", "exact_rotation_angle"],
                )

    def test_prompt_names_the_complete_exact_vocabulary(self) -> None:
        self.assertIn(judge.LEAK_TYPES_JSON, judge.SYSTEM_PROMPT)
        self.assertEqual(
            set(judge.LEAK_TYPE_VOCABULARY),
            {
                "coordinate_pair",
                "exact_correct_map",
                "exact_student_map",
                "exact_translation_value",
                "exact_reflection_line",
                "exact_rotation_angle",
                "other",
            },
        )

    def test_unknown_leak_type_is_rejected(self) -> None:
        value = valid_judgment()
        value["candidate_A"]["forbidden_answer_or_coordinate_leak"] = True
        value["candidate_A"]["leak_types"] = ["exact_reflection_axis"]
        with self.assertRaisesRegex(ValueError, "invalid value"):
            judge.validate_judgment(value, 129)

    def test_invalid_candidate_structure_is_rejected(self) -> None:
        value = valid_judgment()
        del value["candidate_B"]["rationale"]
        with self.assertRaisesRegex(ValueError, "keys must be exactly"):
            judge.validate_judgment(value, 129)

    def test_leak_boolean_must_match_list_emptiness(self) -> None:
        invalid_pairs = (
            (False, ["exact_rotation_angle"]),
            (True, []),
        )
        for leak_boolean, leak_types in invalid_pairs:
            with self.subTest(
                leak_boolean=leak_boolean,
                leak_types=leak_types,
            ):
                value = valid_judgment()
                value["candidate_A"][
                    "forbidden_answer_or_coordinate_leak"
                ] = leak_boolean
                value["candidate_A"]["leak_types"] = leak_types
                with self.assertRaisesRegex(ValueError, "leak boolean"):
                    judge.validate_judgment(value, 129)


class ResumeAndReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = judge.prepare_cases()
        cls.case_by_id = {case["case_id"]: case for case in cls.cases}
        raw_rows = judge.load_jsonl(RAW_EVIDENCE_PATH)
        cls.live_row = next(
            row
            for row in raw_rows
            if row.get("provider") == "tfy"
            and row.get("model") == "gpt-5.6-sol"
        )

    def test_prompt_fix_changes_request_hash_and_run_fingerprint(self) -> None:
        case = self.case_by_id[self.live_row["case_id"]]
        fingerprint = judge.make_run_fingerprint(
            self.cases,
            "tfy",
            "gpt-5.6-sol",
            judge.DEFAULT_TFY_BASE_URL,
            judge.DEFAULT_MAX_OUTPUT_TOKENS,
        )
        self.assertNotEqual(case["request_hash"], self.live_row["request_hash"])
        self.assertNotEqual(fingerprint, self.live_row["run_fingerprint"])
        self.assertNotEqual(
            fingerprint,
            judge.make_run_fingerprint(
                self.cases,
                "tfy",
                "gpt-5.6-sol",
                judge.DEFAULT_TFY_BASE_URL,
                judge.DEFAULT_MAX_OUTPUT_TOKENS,
                system_prompt=judge.SYSTEM_PROMPT + "\nchanged",
            ),
        )

    def test_retry_appends_success_without_rewriting_failed_evidence(self) -> None:
        case = self.case_by_id[self.live_row["case_id"]]
        response_text = self.live_row["attempts"][0]["response_text"]
        fingerprint = judge.make_run_fingerprint(
            self.cases,
            "tfy",
            "gpt-5.6-sol",
            judge.DEFAULT_TFY_BASE_URL,
            judge.DEFAULT_MAX_OUTPUT_TOKENS,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "raw.jsonl"
            original_line = json.dumps(self.live_row, ensure_ascii=False) + "\n"
            raw_path.write_text(original_line, encoding="utf-8")
            with patch.object(judge, "RAW_PATH", raw_path):
                row = judge.call_case(
                    case,
                    lambda _prompt: response_text,
                    "tfy",
                    "gpt-5.6-sol",
                    fingerprint,
                    max_retries=1,
                )
                rows = judge.load_jsonl(raw_path)
                latest = judge.latest_rows()[case["case_id"]]

            self.assertEqual(raw_path.read_text(encoding="utf-8")[: len(original_line)], original_line)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["status"], "failed")
            self.assertEqual(row["status"], "success")
            self.assertEqual(latest["status"], "success")
            self.assertEqual(row["request_hash"], case["request_hash"])
            self.assertNotEqual(row["request_hash"], rows[0]["request_hash"])

    def test_blocked_report_distinguishes_this_run_and_cumulative_attempts(
        self,
    ) -> None:
        metadata = {
            "sample_size": judge.N_PAIRS,
            "provider": "tfy",
            "model": "gpt-5.6-sol",
            "offline_validation": {
                "validated_payloads": judge.SMOKE_PAIRS,
                "payload_case_ids": [129, 552],
            },
        }
        results = {
            "schema_version": judge.SCHEMA_VERSION,
            "status": "BLOCKED",
            "blocked_reason": "persisted smoke failure pending a safe retry",
            "api_calls_made_this_run": 0,
            "offline_validation": metadata["offline_validation"],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            raw_path = temp / "raw.jsonl"
            # Isolate the three preserved failed smoke rows. The live evidence now
            # also contains 100 later successes, which should not turn this blocked-
            # state regression fixture into a successful latest-row state.
            failed_smoke_rows = judge.load_jsonl(RAW_EVIDENCE_PATH)[:3]
            self.assertTrue(all(row["status"] == "failed" for row in failed_smoke_rows))
            raw_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failed_smoke_rows),
                encoding="utf-8",
            )
            paths = {
                "RAW_PATH": raw_path,
                "META_PATH": temp / "metadata.json",
                "RESULTS_PATH": temp / "results.json",
                "REPORT_PATH": temp / "report.md",
            }
            with (
                patch.multiple(judge, **paths),
                patch.dict(
                    os.environ,
                    {"TFY_API_KEY": "", "ANTHROPIC_API_KEY": ""},
                ),
            ):
                judge.write_state(results, metadata)
                report = paths["REPORT_PATH"].read_text(encoding="utf-8")
                written = json.loads(paths["RESULTS_PATH"].read_text(encoding="utf-8"))

        self.assertIn("API calls made in the reported run: **0**", report)
        self.assertIn(
            "cumulative across raw evidence: **9** in **3** raw rows",
            report,
        )
        self.assertIn("--provider tfy --model gpt-5.6-sol --retry-failed", report)
        self.assertEqual(
            written["persisted_attempts_cumulative"]["judge_case_attempts"],
            9,
        )
        self.assertTrue(written["retry_failed_required"])

    def test_offline_dry_run_uses_no_provider_client(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            paths = {
                "OUT_DIR": temp,
                "RAW_PATH": temp / "raw.jsonl",
                "META_PATH": temp / "metadata.json",
                "RESULTS_PATH": temp / "results.json",
                "REPORT_PATH": temp / "report.md",
            }
            with (
                patch.multiple(judge, **paths),
                patch.dict(
                    os.environ,
                    {"TFY_API_KEY": "", "ANTHROPIC_API_KEY": ""},
                ),
                patch.object(
                    judge,
                    "make_tfy_caller",
                    side_effect=AssertionError("dry run constructed a provider client"),
                ),
            ):
                judge.main(
                    [
                        "--dry-run",
                        "--provider",
                        "tfy",
                        "--model",
                        "gpt-5.6-sol",
                    ]
                )
                written = json.loads(paths["RESULTS_PATH"].read_text(encoding="utf-8"))
                report = paths["REPORT_PATH"].read_text(encoding="utf-8")

        self.assertEqual(written["status"], "BLOCKED")
        self.assertEqual(written["api_calls_made_this_run"], 0)
        self.assertNotIn("smoke passed", report.lower())


if __name__ == "__main__":
    unittest.main()
