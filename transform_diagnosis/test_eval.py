"""Tests for the eval harness (`eval.py`).

Verify the harness itself before trusting any number it produces: each binary field
check, the derived ``failure_reason``, semantic transform matching, prediction parsing,
and the aggregate math (esp. balanced accuracy, which must average per-label recall over
only the labels present).

Records are generated in-memory via ``dataset.build_records`` (deterministic, no disk),
and ``chat_format.target_json`` gives each record's gold output — the canonical
"fully-correct prediction" fixture.
"""

from __future__ import annotations

import json

from transform_diagnosis import chat_format, dataset
from transform_diagnosis import eval as ev
from transform_diagnosis import transform_core as tc

SEED = 20260708
N = 200
MIN_COUNT = 20
OOD = 12


def _records():
    recs, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    return recs


def _one(label: str) -> dict:
    """First in-distribution record carrying ``label`` (falls back to any split)."""
    recs = _records()
    for r in recs:
        if r["label"] == label:
            return r
    raise AssertionError(f"no record with label {label!r}")


# --------------------------------------------------------------------------------------
# The gold output scores a perfect pass — for every label. Gold hints are now
# coordinate-free Socratic nudges, so hint_ok (family + no leak) passes while the strict
# hint_exact_ok does NOT (the exact-token contract was the leak and has been removed).
# --------------------------------------------------------------------------------------

def test_gold_output_scores_all_pass_for_every_label():
    recs = _records()
    seen = set()
    for rec in recs:
        if rec["label"] in seen:
            continue
        seen.add(rec["label"])
        row = ev.score_record(chat_format.target_json(rec), rec)
        assert row["parse_ok"] and row["label_ok"], rec["label"]
        assert row["transform_ok"], rec["label"]
        assert row["hint_ok"], (rec["label"], rec["hint"])
        assert row["failure_reason"] == "", rec["label"]
    assert seen == set(tc.DIAGNOSIS_LABELS)


def test_gold_completely_wrong_hint_is_not_flagged_as_leak():
    # The completely_wrong gold hint names the translation family ("slide") but states no
    # (dx, dy) vector or map literal, so it passes hint_ok and is not flagged as a leak.
    rec = _one("completely_wrong")
    row = ev.score_record(chat_format.target_json(rec), rec)
    assert row["hint_ok"] and row["failure_reason"] == "", rec["hint"]


# --------------------------------------------------------------------------------------
# Individual failure modes
# --------------------------------------------------------------------------------------

def test_wrong_label():
    rec = _one("wrong_translation")
    pred = {"label": "wrong_rotation_angle", "correct_transform": rec["correct_transform"],
            "hint": rec["hint"]}
    row = ev.score_record(json.dumps(pred), rec)
    assert row["parse_ok"] and not row["label_ok"]
    assert row["pred_label"] == "wrong_rotation_angle"
    assert row["failure_reason"] == "wrong_label:wrong_translation->wrong_rotation_angle"


def test_malformed_json_is_parse_fail():
    rec = _one("correct")
    row = ev.score_record("sorry, I think the answer is correct!", rec)
    assert not row["parse_ok"]
    assert row["pred_label"] == "PARSE_FAIL"
    assert not (row["label_ok"] or row["transform_ok"] or row["hint_ok"])
    assert row["failure_reason"] == "parse_fail"


def test_unknown_label_is_parse_fail():
    rec = _one("correct")
    row = ev.score_record(json.dumps({"label": "banana", "correct_transform": [], "hint": ""}), rec)
    assert not row["parse_ok"] and row["pred_label"] == "PARSE_FAIL"


def test_hint_missing_token():
    rec = _one("wrong_reflection_line")
    pred = {"label": rec["label"], "correct_transform": rec["correct_transform"],
            "hint": "You made a mistake somewhere."}
    row = ev.score_record(json.dumps(pred), rec)
    assert row["label_ok"] and row["transform_ok"]
    assert not row["hint_ok"]
    assert row["failure_reason"] == "hint_missing_token"


def test_hint_leak_of_extra_coordinates():
    rec = _one("wrong_reflection_line")
    # Otherwise-correct hint, but it states a coordinate pair it was never sanctioned to.
    pred = {"label": rec["label"], "correct_transform": rec["correct_transform"],
            "hint": rec["hint"] + " The vertex should land at (999, -999)."}
    row = ev.score_record(json.dumps(pred), rec)
    assert row["label_ok"] and row["transform_ok"]
    assert not row["hint_ok"]
    assert row["failure_reason"] == "hint_leak"


# --------------------------------------------------------------------------------------
# Semantic transform matching (wording-invariant) + parsing robustness
# --------------------------------------------------------------------------------------

def test_transform_match_is_semantic_not_string():
    gold = ["rotate 90 degrees counterclockwise", "translate 7 right"]
    # Same motion, different sanctioned wording (cw complement) -> still a match.
    variant = ["rotate 270 degrees clockwise", "translate 7 right"]
    assert ev._transform_match(variant, gold)
    # A genuinely different motion is not a match.
    assert not ev._transform_match(["rotate 180 degrees counterclockwise", "translate 7 right"], gold)


def test_transform_match_rejects_unparseable_and_wrong_length():
    gold = ["reflect across x axis", "translate 3 up"]
    assert not ev._transform_match(["reflect across x axis"], gold)  # wrong length
    assert not ev._transform_match("not a transform", gold)          # unparseable single str


def test_transform_match_accepts_previously_brittle_phrasings():
    # These are exactly the natural phrasings that used to fall through to the brittle
    # exact-string fallback and score transform_ok=false despite being correct.
    # (rotation with no direction + "about the origin"; "units" noun in a translation.)
    gold_180 = ["rotate 180 degrees counterclockwise", "translate 5 up"]
    assert ev._transform_match(["rotate 180 degrees about the origin", "translate 5 up"], gold_180)
    assert ev._transform_match(["rotate 180 degrees counterclockwise", "translate 5 units up"], gold_180)

    gold_refl = ["reflect across x axis", "translate 2 left"]
    assert ev._transform_match(["reflect across the x-axis", "translate 2 units left"], gold_refl)

    # Guard against over-looseness: a genuinely different motion still fails, and an
    # ambiguous no-direction 90 rotation does NOT get silently accepted as the gold.
    assert not ev._transform_match(["rotate 90 degrees about the origin", "translate 5 up"], gold_180)
    assert not ev._transform_match(["rotate 180 degrees about the origin", "translate 5 down"], gold_180)


# --------------------------------------------------------------------------------------
# Hint metric — the achievable, instruction-consistent check (operation family + no leak).
# A hint that OBEYS "point at the error WITHOUT stating the coordinates" must be able to
# pass, including for translation labels (whose exact canonical token IS the answer).
# --------------------------------------------------------------------------------------

def test_instruction_following_translation_hint_can_pass():
    rec = _one("opposite_translation")
    # References the translation/direction concept, states NO coordinates -> should pass
    # the primary metric even though it does NOT contain the exact "translate N dir" token.
    hint = ("Your reflection looks right; now recheck the direction of the final slide — "
            "same distance, but is it going the opposite way?")
    pred = {"label": rec["label"], "correct_transform": rec["correct_transform"], "hint": hint}
    row = ev.score_record(json.dumps(pred), rec)
    assert row["label_ok"] and row["transform_ok"]
    assert row["hint_ok"], hint
    assert row["failure_reason"] == ""
    # The strict/secondary metric legitimately still fails (no exact token), which is why
    # it is exploratory-only and not the headline number.
    assert not row["hint_exact_ok"]


def test_wrong_translation_hint_family_required():
    rec = _one("wrong_translation")
    good = {"label": rec["label"], "correct_transform": rec["correct_transform"],
            "hint": "Your flip is fine; compare how far one vertex must slide — your shift is off."}
    assert ev.score_record(json.dumps(good), rec)["hint_ok"]


def test_bad_and_empty_hints_still_fail():
    rec = _one("opposite_translation")
    for bad in ("You made a mistake somewhere.", "", "Check your rotation angle."):
        pred = {"label": rec["label"], "correct_transform": rec["correct_transform"], "hint": bad}
        row = ev.score_record(json.dumps(pred), rec)
        assert not row["hint_ok"], bad
        assert row["failure_reason"] == "hint_missing_token", bad


def test_hint_family_metric_matches_gold_and_leak_guard():
    # The gold hint passes the family metric (hint_ok) but, being coordinate-free, no
    # longer contains the exact tokens, so the strict hint_exact_ok is False. The
    # coordinate-leak guard still fires on unsanctioned (x, y) pairs appended to it.
    rec = _one("wrong_reflection_line")
    gold = ev.score_record(chat_format.target_json(rec), rec)
    assert gold["hint_ok"] and not gold["hint_exact_ok"]
    leaky = {"label": rec["label"], "correct_transform": rec["correct_transform"],
             "hint": rec["hint"] + " It should land at (999, -999)."}
    row = ev.score_record(json.dumps(leaky), rec)
    assert not row["hint_ok"] and row["failure_reason"] == "hint_leak"


def test_expected_hint_families_are_operation_concepts():
    from transform_diagnosis import hints
    rt = _one("opposite_translation")
    assert hints.expected_hint_families("opposite_translation", rt) == ["translation"]
    rr = _one("wrong_reflection_line")
    assert hints.expected_hint_families("wrong_reflection_line", rr) == ["reflection"]
    ra = _one("wrong_rotation_angle")
    assert hints.expected_hint_families("wrong_rotation_angle", ra) == ["rotation"]


def test_parse_pred_tolerates_fences_and_prose():
    obj = {"label": "correct", "correct_transform": ["identity", "identity"], "hint": "ok"}
    fenced = "```json\n" + json.dumps(obj) + "\n```"
    assert ev.parse_pred(fenced) == obj
    prosey = "Here is my answer: " + json.dumps(obj) + " -- done."
    assert ev.parse_pred(prosey) == obj
    assert ev.parse_pred("no json here") is None


# --------------------------------------------------------------------------------------
# Chain-of-thought outputs: a reasoning PREFIX before the final JSON must not disturb
# scoring — parse_pred pulls the LAST JSON object, so "reasoning ... {JSON}" scores exactly
# as the bare JSON would. (Guards the CoT fine-tune eval path; the JSON schema is unchanged.)
# --------------------------------------------------------------------------------------

def test_parse_pred_recovers_final_json_after_reasoning_prefix():
    obj = {"label": "wrong_translation",
           "correct_transform": ["translate 1 down", "rotate 180 degrees counterclockwise"],
           "hint": "Check the translation."}
    reasoning = (
        "The intended transformation maps the RED pre-image onto the GREEN correct image: "
        "first translate 1 down, then rotate 180 degrees counterclockwise.\n"
        "The student's BLUE answer corresponds to: first translate by (1, 4), then rotate "
        "180 degrees counterclockwise.\n"
        "The second step is correct, but the first step is wrong. So the diagnosis is "
        "wrong_translation.\n"
    )
    assert ev.parse_pred(reasoning + json.dumps(obj)) == obj
    # A fenced final answer after the reasoning is handled too (fence is outside the braces).
    fenced_tail = reasoning + "```json\n" + json.dumps(obj) + "\n```"
    assert ev.parse_pred(fenced_tail) == obj


def test_parse_pred_takes_last_object_when_multiple():
    first = {"label": "correct", "correct_transform": ["identity", "identity"], "hint": "a"}
    last = {"label": "wrong_rotation_angle",
            "correct_transform": ["rotate 90 degrees counterclockwise", "translate 2 up"],
            "hint": "b"}
    # A decoy JSON object earlier in the text must NOT win over the final answer.
    text = "First I considered " + json.dumps(first) + " but then:\n" + json.dumps(last)
    assert ev.parse_pred(text) == last


def test_reasoning_prefix_scores_identically_to_bare_json():
    # A real, gold-consistent CoT target (trace + JSON) must score exactly like its JSON.
    from transform_diagnosis import cot
    for label in tc.DIAGNOSIS_LABELS:
        rec = _one(label)
        bare = ev.score_record(chat_format.target_json(rec), rec)
        withcot = ev.score_record(cot.cot_target(rec), rec)
        for field in ("parse_ok", "label_ok", "transform_ok", "hint_ok",
                      "hint_exact_ok", "pred_label", "failure_reason"):
            assert bare[field] == withcot[field], (label, field, bare[field], withcot[field])


# --------------------------------------------------------------------------------------
# Aggregation math
# --------------------------------------------------------------------------------------

def _row(true_label, pred_label, label_ok, parse_ok=True, transform_ok=True, hint_ok=True):
    return {"true_label": true_label, "pred_label": pred_label, "parse_ok": parse_ok,
            "label_ok": label_ok, "transform_ok": transform_ok, "hint_ok": hint_ok}


def test_balanced_accuracy_averages_per_label_recall_over_present_labels():
    # correct: 3/3 right (recall 1.0); wrong_translation: 0/1 right (recall 0.0).
    results = [
        _row("correct", "correct", True),
        _row("correct", "correct", True),
        _row("correct", "correct", True),
        _row("wrong_translation", "correct", False),
    ]
    agg = ev.aggregate(results)
    assert agg["n"] == 4
    assert abs(agg["label_accuracy"] - 0.75) < 1e-9          # raw accuracy
    assert abs(agg["balanced_accuracy"] - 0.5) < 1e-9        # (1.0 + 0.0) / 2, present only
    assert agg["per_label_recall"]["correct"] == 1.0
    assert agg["per_label_recall"]["wrong_translation"] == 0.0
    assert agg["per_label_recall"]["completely_wrong"] is None  # absent -> None, excluded
    # confusion: one wrong_translation record predicted as correct
    assert agg["confusion"]["wrong_translation"]["correct"] == 1
    assert agg["confusion"]["correct"]["correct"] == 3


def test_parse_fail_shows_in_confusion_column():
    results = [_row("correct", "PARSE_FAIL", False, parse_ok=False,
                    transform_ok=False, hint_ok=False)]
    agg = ev.aggregate(results)
    assert agg["parse_rate"] == 0.0
    assert agg["confusion"]["correct"]["PARSE_FAIL"] == 1


def test_reporting_helpers_smoke():
    base = ev.aggregate([_row("correct", "wrong_translation", False)])
    tuned = ev.aggregate([_row("correct", "correct", True)])
    table = ev.format_table(base, tuned)
    assert "balanced_acc" in table and "delta" in table
    conf = ev.format_confusion(tuned)
    assert "true\\pred" in conf


def test_save_results_writes_agg_and_per_record_jsonl(tmp_path):
    rec = _one("correct")
    rows = [ev.score_record(chat_format.target_json(rec), rec)]
    agg = ev.aggregate(rows)
    agg_path = tmp_path / "results.json"
    rec_path = tmp_path / "records.jsonl"
    ev.save_results(agg, str(agg_path), rows, str(rec_path))
    assert json.loads(agg_path.read_text())["n"] == 1
    saved = [json.loads(l) for l in rec_path.read_text().splitlines() if l.strip()]
    assert len(saved) == 1
    assert list(saved[0].keys()) == list(ev.RECORD_FIELDS)
