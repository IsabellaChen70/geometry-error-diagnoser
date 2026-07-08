No further tool calls are needed here — this is a synthesis task over the provided research. Producing the final report directly.

## Datasets found, by angle

**Angle 1 — Education / misconception / tutoring-dialogue**
- Eedi/Question-Anchored-Tutoring-Dialogues-2k — huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k — UK K-12 math tutor-student dialogues anchored to diagnostic questions, "talk move" annotations. 1,971 interventions/68,717 msgs. CC-BY-NC-4.0.
- eth-nlped/mathdial (+ mirrors: mathdial-chat, MathDial-*, eth-mathdial-conversations) — huggingface.co/datasets/eth-nlped/mathdial — 7th-grade math word-problem tutoring dialogues with typed confusion categories (relevance/procedural/equation-ordering) and dialogue acts. 2,861 rows, CC-BY-4.0.
- nanote/algebra_misconceptions ("MaE") — huggingface.co/datasets/nanote/algebra_misconceptions — 55 named middle-school algebra misconceptions x 4 examples each (220 core items, 1,130 rows). MIT. GPT-4 reported at 83.9% misconception-ID accuracy, worse on ratios/proportions.
- guldasta/Math_misconception — huggingface.co/datasets/guldasta/Math_misconception — fractions-only, 15 questions x 35 misconception types, 36,696 rows, license unspecified.
- ShethArihant/eedi-train-subset (+ mirrors: rbiswasfc/eedi-awq-calibration*, VaggP/Eedi-competition-kaggle-*, cdtmc/eedi-ir) — huggingface.co/datasets/ShethArihant/eedi-train-subset — Kaggle "Eedi – Mining Misconceptions in Mathematics" mirror; ~1,857 base questions expanded to 109K+ rows, expert-written wrong-answer→MisconceptionId/Name taxonomy. License unspecified (check Kaggle terms).
- rik1599/eedi-response-data — huggingface.co/datasets/rik1599/eedi-response-data — algebra/number/geometry splits, ~1.38M rows of real student response distributions (not misconception-labeled itself). License unspecified.
- vancevo/misconception_mining / misconception_mining_asag — huggingface.co/datasets/vancevo/misconception_mining — thin documentation, 10,000 rows, likely an Eedi derivative. License unspecified.
- Meyerger/ASAG2024 — huggingface.co/datasets/Meyerger/ASAG2024 — composite short-answer grading (SciEntsBank science, SAF, BEETLE II physics/circuits, plus CS/stats subsets to exclude). 56,646 rows, continuous 0-100 grade, mixed licenses.
- knght0wl21/socratic-tutoring-dataset — huggingface.co/datasets/knght0wl21/socratic-tutoring-dataset — graduate-level math (real analysis/topology/abstract algebra), not K-12. 195 rows. License unspecified.
- HISTORYQUIZ2024/raw_dataset_10-12 (+ "_advanced") — huggingface.co/datasets/HISTORYQUIZ2024/raw_dataset_10 — raw Vietnamese history lesson text, no QA/misconception layer, ~200-290 rows each.
- Viswa45/ASAG (Mohler benchmark) — CS-101 short-answer grading. **Excluded (CS).**
- SciEntsBank/BEETLE II subsets — science ASAG; bio/chem-tagged portions **excluded**, physics/circuits portion in-scope.
- Notable exclusions mentioned: Senthil-HIGS/FEEDBACK_BASED_SOURCE_CODE_GENERATION (CS), Lots-of-LoRAs/task081_piqa_wrong_answer_generation (not curriculum), CS/stats subsets in ASAG2024 (CS).

**Angle 2 — Structured-output / strict-JSON / function-calling**
- epfl-dlab/JSONSchemaBench — huggingface.co/datasets/epfl-dlab/JSONSchemaBench — GitHub/K8s/Snowplow/WashingtonPost/Glaive schemas, ~19,100 rows, MIT. Generic, no subject.
- glaiveai/glaive-function-calling-v2 — generic assistant tool-use, ~112K, Apache-2.0.
- NousResearch/hermes-function-calling-v1 — IoT/quantum/hospitality SaaS tool-use, 1,890 rows (main), Apache-2.0.
- Salesforce/xlam-function-calling-60k — 3,673 APIs/21 categories, 60,000 rows, CC-BY-4.0 (gated).
- hypervariance/function-calling-sharegpt — generic, 86,864 rows, Apache-2.0.
- nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2 — synthetic multi-topic (pet care, wine, science-fair, etc. as filler), 62,696 rows, CC-BY-4.0.
- mdonigian/json-schema-compliance-benchmark — deliberately novel/absurd hobbyist domains (anti-contamination design), 500 rows, Apache-2.0.
- mdonigian/synthetic-structured-output-dataset — generic, 10K-100K, MIT.
- vericava/sft-tool-calling-structured-output-v1 — generic EN/JA, 100K-1M, Apache-2.0.
- amphora/FC-Text-to-JSON-150k — generic function-calling reasoning traces, 151,519 rows.
- fireworks-ai/function-calling-eval-dataset-v0 — generic, <1K rows.
- dataunitylab/json-schema (+ json-schema-store/-descriptions/-definitions/-keywords) — raw GitHub schema corpora, 1K-100K each, mixed license.
- achinta3/cybersec-jsonschemabench-* (6 variants) — AWS CloudTrail/incident logs, 1K-10K, long-context.
- philipp-zettl/german-structured-output — NER/RE/function-calling, German, 1K-10K, CC-BY-SA-4.0.
- agentlans/text-to-json — generic text→JSON, 10K-100K, CC-BY-4.0.
- Adjacent K-12/AP subject data with NO structured-output format: tasksource/AES2-essay-scoring (17,307 essays, CSV, CC-BY-NC-4.0, scalar 1-6 score only), jatinmehra/Automated-Essay-Scoring-2.0, DysfunctionalHuman/essay-scoring, abdlh/...24_textual_features (same lineage), zipu-w/AP-exam-questions (AP Calc/CS/Econ/Physics/Psych, <1K rows, plain Q&A), LindaHuang/ap-exam-questions (minimal metadata).
- **Key finding: no dataset anywhere pairs strict-JSON/schema output with real K-12/AP subject content** — this intersection is empty on the Hub.

**Angle 3 — Logic puzzle / abstract reasoning**
- ARC-AGI (lordspline/arc-agi, dataartist/arc-agi, arcprize/arc_agi_v1_public_eval, mertaylin/arc-agi*) — grid pattern transformation, 400-100K+ rows, MIT. Pure abstract, same flavor as the already-rejected candidate.
- Zebra puzzles (carbonteq/rg-zebra_puzzles-instruct-100k, dhruveshpatel/zebra-puzzle, TTTXXX01/Puzzle_Zebra*) — constraint-satisfaction deduction, 100K train/4K test (carbonteq), MIT. Abstract as shipped; generator is content-swappable.
- Sudoku family (sapientinc/sudoku-extreme, abatilo/sudokubench, Ritvik19/Sudoku-Dataset, AIML-TUDA/CLEVR-Sudoku) — 1.29M-4.2M rows (standard), CLEVR-Sudoku 6,000 rows/CC-BY-4.0 adds visual attribute parsing.
- Chess spatial reasoning (oscar128372/chess_spatial_reasoning_400kv1, _10k) — FEN/PGN board reasoning, 10K-400K rows, license unspecified.
- eousphoros/2d_3d_seq_path_spatial_reasoning — Hamiltonian path/parity-proof grids, 1,856 rows, MIT. Card itself states no curriculum tie.
- Ashima/*constraint_satisfaction* — generic CSPs, 500-1K rows, thin/low-quality, license unclear.
- Spatial reasoning grab-bag (juletxara/visual-spatial-reasoning, ReasonCore/open-spatial-reasoning) — mostly VQA/scene-grounding, off-topic for puzzle logic.
- theblackcat102/syllogism, VietGPT-AI/sft_syllogism — categorical syllogism validity, 1,330 to 10K-100K rows, license unspecified. Best clean curriculum tie (geometry deductive-proof unit, ELA argumentation).

**Angle 4 — Subject diagrams beyond geometry**
- derek-thomas/ScienceQA — huggingface.co/datasets/derek-thomas/ScienceQA — multi-subject (earth science, non-circuit physics, geography/civics/economics, + bio/chem to filter out), ~21,200 rows, CC-BY-SA-4.0.
- lmms-lab/ai2d (+ mirrors Ryoo72/ai2d, llamastack/ai2d) — elementary/middle-school diagrams (food chains, water cycle, geology, simple physics), ~3,088 test rows, research-use license.
- gsjang/ScholarBench_MC_earth_life_sciences — Korean earth/life-science MC, <1K rows, text-only, license unspecified. Weak fit.
- AshkanTaghipour/mineral-exploration-geology-qa — professional geology QA, text-only, <1K, Apache-2.0. Not K-12.
- ibrahimatlgn/bangka-1930s-topographic-maps — raw scanned maps, no QA layer, <1K images, MIT.
- Jackrong/financial-economics-reasoning — text-only finance/econ reasoning, 100K-1M rows, Apache-2.0. Not diagram-based.
- gankun/G-Social-Science-Diagram — synthetic GPT-generated social-science diagrams, 495 rows, license unclear.
- wfzimmerman/wwii-naval-armament, /fleets-of-wwii-design-commentary — WWII naval history tabular specs, <1K-10K rows, CC-BY-NC-4.0, no images.
- ZheqiDAI/MusicScore — sheet-music images (IMSLP), 403/14,656/204,800-image tiers, CC license, no QA layer.
- Sweaterdog/music-theory-images-20k — explicit VQA-shaped music theory, 10K-100K rows, Apache-2.0. Best music-theory fit found.
- m-a-p/MusicTheoryBench — music theory benchmark, <1K rows, text-only (likely), CC license.
- HumynLabs/physics-problems — general physics, image modality tagged, <1K rows, CC-BY-4.0, subtopics unclear.
- veggiebird/physics-scienceqa / AnonySub628/physics-scienceqa — physics-filtered ScienceQA slice, text-only (image rows apparently excluded).
- achang/plot_qa (+ mirrors martinsinnona/plotqa, DanhVuiVe/PlotQa_clean, nimapourjafar/mm_plotqa) — chart/plot VQA, ~224,377 plots, 100K-1M rows in this mirror, CC license.
- ReadingTimeMachine/visual_qa_histograms, YifeiDevs/scatter-plot — narrow single-chart-type sets, 1K-10K rows, Apache-2.0.
- Confirmed gaps: no optics-diagram, free-body-diagram, AP History timeline/map/primary-source-image, or AP Government/Economics institutional-diagram datasets found.

**Angle 5 — Text-only reasoning-gap tied to a subject**
- tasksource/logical-fallacy — 13 fallacy types incl. an "edu" config, 3,761 rows, license unspecified. Flat classification, not curriculum-framed.
- Navy0067/contrastive-pairs-for-logical-fallacy — 703 contrastive valid/fallacious pairs, CC-BY-4.0. Generic, tiny.
- strickvl/counterfactual_history_reasoning — huggingface.co/datasets/strickvl/counterfactual_history_reasoning — AP US/World History topics (WWII, Civil Rights, Industrial Revolution, Cold War, etc.), counterfactual premise/reasoning-trace/conclusion structure, 100 rows, MIT. DeepSeek-R1-generated traces (unverified quality).
- syrgkanislab/CausalReasoningBenchmark — causal inference/econometrics grounded in 85 real peer-reviewed papers, 174 rows (542MB w/ supporting data), license unspecified. Graduate-level, not K-12.
- tasksource/counterfactually-augmented-snli, /counterfactually-augmented-imdb — generic NLI/sentiment, no subject anchor. Methodological template only.
- DriftLogic/Annotated_Persuasive_Essays — argument-mining (claims/premises/support-attack graph), free sample only 5 rows, full 150-essay set paywalled, CC-BY-NC-4.0.
- Svngoku/African-History-Extra-11-30-24-QA-Pairs-Reasoning (+ african-history-qa-reasoning) — African history 7th-20th century, <1K rows, MIT. General explanatory QA, not deliberately adversarial.
- EdmondFU/Causal-Reasoning-Bench_CRBench — generic CoT causal-error correction, 10K-100K rows, no subject anchor.
- Confirmed near-total gap: DBQ, AP Bio experimental design, physics misconception, SAT reading, rhetorical device, combinatorics/case-splitting, ambiguity resolution all returned zero or off-topic results.

**Angle 6 — Broad trending sweep**
- AI-for-Education/pedagogy-benchmark — Chilean teacher-training exam MCQs (pedagogy, not subject content), 1,143 rows, MIT.
- HolySaint/MBE-exam-questions (+ bsharve mirror) — US Bar Exam questions w/ IRAC framework, 10K-100K rows, MIT. Not K-12.
- geekyrakshit/indian-exam-questions — NEET physics/chemistry with figures + solutions, ~180 rows/subset. Physics portion in-scope; chemistry excluded.
- AI-MO/olympiads — raw PDF/markdown/JSONL olympiad archive, 3.24GB, algebra/geometry/combinatorics/etc.
- Metaskepsis/Olympiads_hard/_medium/Olympiads — difficulty-stratified NuminaMath-CoT subsets, 32,926/12,893/20,672 rows, MIT.
- TuringEnterprises/Rubric-Graded-Reasoning — 150 PhD tasks (CS/data-science/chemistry — excluded domains), rubric-grading design pattern only.
- Tushe/tushe-grade-school-stem — South African CAPS-aligned STEM textbooks (grades 4-12), 70.3MB, CC-BY-4.0, raw text not QA.
- dry-melon/Chinese-middle-school-English-exam-questions — grades 7-9 EFL cloze/fill-in-blank w/ "test point" skill tags, 35.2K rows, CC-BY-4.0.
- zipu-w/alevel-exam-questions (+ AP-exam-questions, tmua-exam-questions-with-images) — UK A-level Accounting/Chemistry/Economics/Physics w/ images, 480 rows.
- wuulong/purchasing_exam_questions — Taiwan procurement law exam, 3,695 rows, CC-BY-SA-4.0. Not curriculum-relevant.
- tejeshbhalladhanyog/sa-finance-reasoning-mix — financial reasoning, 288,923 rows, Apache-2.0. Professional, not K-12.
- nvidia/PhysicalAI-Traffic-Anomaly-Reasoning — video causal/temporal reasoning, 44,040 annotations, CC-BY-4.0. Different modality, not curriculum.
- Honorable mentions (unverified/gated/excluded): K12_Testing_Questions (DataoceanAI, gated 401), Kun-Xiang/ViRL39K-GradeSchool, JTBTechnology/tcm_exam_questions (bio/medicine, excluded), Roman1111111/gemini-3.1-pro-hard-high-reasoning sets.

## Ranked topic shortlist

**1. Math misconception diagnosis from wrong MCQ answer (Eedi taxonomy) — TOP PICK**
- Subject: K-12 math (UK curriculum, maps well to US middle/early-HS algebra and number sense). Curriculum-grounded via the actual Eedi/Kaggle competition's expert-written misconception taxonomy, not an invented puzzle.
- Datasets: ShethArihant/eedi-train-subset (huggingface.co/datasets/ShethArihant/eedi-train-subset) as the core question+wrong-answer+MisconceptionId/Name corpus; optionally cross with rik1599/eedi-response-data for realistic answer-distribution grounding, and nanote/algebra_misconceptions for a small, cleanly-curated eval slice.
- Litmus test evidence: nanote/algebra_misconceptions reports GPT-4 at only 83.9% accuracy identifying misconceptions (worse on ratios/proportions) — this is one of the only candidates in the entire research with a *documented* base-model failure rate, not just a plausible guess. The Kaggle competition itself was run precisely because this task is hard for models.
- Scope narrowness: Very narrow if restricted to one misconception family (e.g., just "ratio/proportion" or just "negative number" misconceptions) rather than the full ~200-misconception taxonomy — recommend picking one Eedi topic cluster, not the whole competition.
- Note: license on the Eedi mirrors is unspecified; verify Kaggle competition terms before committing.

**2. Fraction-simplification misconception classification**
- Subject: K-12 math, fractions unit specifically (simplification, visual/shaded-region representations) — a single, well-defined curriculum topic taught explicitly in grades 3-6.
- Datasets: guldasta/Math_misconception (huggingface.co/datasets/guldasta/Math_misconception) — 15 questions x 35 misconception types, 36,696 rows.
- Litmus test evidence: Needs verification — the dataset card doesn't report a base-model benchmark; it's a plausible candidate (fraction misconceptions are notoriously persistent and subtle) but unlike candidate #1 there's no cited AI-failure number.
- Scope narrowness: Excellent — single topic, single question format (MCQ + explanation), categorical label makes grading trivial (auto-verifiable by construction). Possibly the single narrowest, cleanest-scoped candidate in the whole set.
- Caveat: license unspecified; 15 unique underlying questions means real diversity is thin despite 36,696 rows (heavy duplication/resampling) — would need augmentation for a real SFT set, though fine as a seed/eval.

**3. Math-dialogue confusion-type classification + "probe not tell" tutoring move**
- Subject: 7th-grade math word problems — explicit, well-defined grade-level curriculum tie.
- Datasets: eth-nlped/mathdial (huggingface.co/datasets/eth-nlped/mathdial) — 2,861 rows, typed confusion categories (relevance/procedural/equation-ordering) + dialogue-act labels (probing/generic/focus/telling).
- Litmus test evidence: Not directly benchmarked in the card as provided — flagged as "needs verification." Plausible failure mode: models default to "telling" (revealing the answer/procedure) rather than "probing," which is exactly the behavior your project wants to instill (diagnose without revealing the answer), so the dataset's own act-label taxonomy is unusually well-aligned to your target behavior even without a cited failure rate.
- Scope narrowness: Good — could narrow further to just "equation-ordering confusion" as the one target misconception type.
- Strength: this is the candidate structurally closest to what you're already building (diagnose-and-withhold-answer), just swapping geometry for word-problem algebra.

**4. Music theory notation reading (VQA)**
- Subject: Music theory — a real, standard K-12/AP curriculum (AP Music Theory exists as a College Board AP course with notation-reading as a core skill), genuinely under-served territory (nobody else is likely targeting this).
- Datasets: Sweaterdog/music-theory-images-20k (huggingface.co/datasets/Sweaterdog/music-theory-images-20k) — 10K-100K rows, explicitly VQA-shaped, Apache-2.0.
- Litmus test evidence: Not verified — no benchmark numbers in the card; needs a manual read of question types (note names, key signatures, interval identification, rhythm counting) to confirm both curriculum-depth and that a well-prompted base model actually struggles (score-reading VQA might be easy for strong multimodal base models, unlike text-based misconception diagnosis).
- Scope narrowness: Would need tight narrowing — e.g., restrict to "identify the specific rhythmic/interval-counting error in this measure" rather than all of music theory. Promising precisely because it's not math and not geometry, but the least-verified candidate here on the "AI already fails" criterion.
- Risk: this is a genuinely different modality (image) from your current geometry candidate, which may or may not be in scope depending on whether you want to stay text-only.

**5. Syllogism / deductive-proof validity checking (geometry's if-then unit)**
- Subject: Geometry's deductive-proof unit and/or ELA argumentation standards — syllogistic form ("if a shape is a square, it is a rectangle...") is explicitly taught curriculum content, not an abstract IQ puzzle.
- Datasets: theblackcat102/syllogism (difficulty-leveled, 1,330 rows) and VietGPT-AI/sft_syllogism (10K-100K rows, legal-domain skew).
- Litmus test evidence: Needs verification — no cited base-model failure rate in either card. Syllogism validity-checking is a well-known classic reasoning-gap category in the broader NLP literature (not cited in this specific research batch), so plausible but unconfirmed here.
- Scope narrowness: Good if restricted to one syllogism form (e.g., categorical syllogisms with a single fixed structure) mapped explicitly to geometry's proof-writing unit rather than generic logic.
- Caution: risk of drifting back toward "abstract puzzle with thin subject tie" — the exact failure mode of the previously-rejected candidate — unless you anchor every example explicitly in a specific geometry theorem/proof-step vocabulary, not generic all-purpose syllogisms.

**6. Chart/plot misreading in AP Statistics**
- Subject: AP Statistics — chart/graph interpretation (reading values, comparing series, identifying misleading axes/scales) is core, explicitly examined curriculum content.
- Datasets: achang/plot_qa (huggingface.co/datasets/achang/plot_qa, 100K-1M rows in this mirror) as a base to subset/relabel, potentially combined with ReadingTimeMachine/visual_qa_histograms for a single-chart-type narrow slice.
- Litmus test evidence: Explicitly flagged in the research as weak — "many PlotQA questions are simple value lookups that models solve easily." This is a real risk factor: the raw dataset does NOT already demonstrate model failure, and would need hand-selection of the harder question subtype (e.g., trend extrapolation or misleading-scale detection) to have any chance of passing the litmus test.
- Scope narrowness: Would require significant curation work to get to "one target type of statistical misreading" — not off-the-shelf narrow.
- Verdict: weaker than 1-3, but included because it's genuinely AP-curriculum-tied (unlike geometry-transformation-composition) and chart literacy is a real, well-known area where casual model use produces errors.

**7. AP History counterfactual/causal reasoning**
- Subject: AP US/World History (WWII, Civil Rights Movement, Industrial Revolution, Cold War, Renaissance, Space Race) — real AP-tested content areas.
- Datasets: strickvl/counterfactual_history_reasoning (huggingface.co/datasets/strickvl/counterfactual_history_reasoning) — 100 rows, MIT.
- Litmus test evidence: None provided — reasoning traces are themselves DeepSeek-R1-generated and unverified for correctness, so you'd be bootstrapping ground truth rather than inheriting it. This is the weakest evidentiary base of any candidate on this list.
- Scope narrowness: Interesting shape (premise/reasoning-trace/conclusion) but "counterfactual history" is inherently open-ended/non-auto-verifiable — hard to get strict, checkable JSON ground truth out of "what if the Cold War had ended differently," which cuts against your auto-verifiable-ish requirement.
- Verdict: include only if you're willing to do substantial dataset construction; not recommended as a primary pick given the ground-truth verifiability problem.

## Honest note on what this search can and cannot tell you

A Hugging Face dataset search can reliably tell you whether curriculum-grounded content *exists in some form* and roughly how it's shaped (size, format, license, labeling scheme) — but existence is not evidence of AI difficulty. Only two candidates in this entire six-angle sweep came back with an actual reported model-failure number baked into the dataset's own documentation: nanote/algebra_misconceptions (GPT-4 at 83.9% on misconception identification, worse on ratios/proportions — supports candidate #1/#2's family) and, tangentially, the general observation that the Eedi Kaggle competition itself was run because leaderboard models struggled at the task. Everything else — mathdial's confusion-type/dialogue-act framing, the syllogism sets, the music-theory VQA set, and the PlotQA-based statistics idea — is "a relevant, curriculum-grounded dataset exists" with the AI-difficulty claim unverified; for PlotQA the research explicitly flags evidence pointing the *other* way (many items are easy lookups). Before committing a full week to any candidate other than #1/#2, you should run your own quick litmus test — take 20-30 real examples from the chosen dataset and try a well-prompted frontier base model on them directly — because the absence of a documented failure rate in a dataset card just means nobody measured it, not that the model succeeds.