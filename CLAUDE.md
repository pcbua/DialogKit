# DialogKit: notes for Claude

Read this before working here. It records how we work and the decisions behind the benchmark, so a new session does
not redo or undo them. The user's own docs: `README.md`, `CONTRIBUTING.md`, `docs/index.md` (tools and how to run them).

## How we work
- **GitHub Flow, `main` is protected** (ruleset: pull request required, no force push). Claude creates the branch, commits, pushes and **opens the PR itself**; the user reviews and merges. Never hand PR creation back to the user.
- **Before branching: `git fetch origin` and branch from `origin/main`.** Before running a script whose results go into the repo, be on current `origin/main`. Before pushing to an existing PR branch, check that the PR is still open (the user merges fast).
- Small, focused PRs; one PR may hold the code commit and the results commit of one change. Commit messages and PR descriptions say what and why.
- Tests: `uv run pytest` (also in CI on every PR). Every `utils/<script>.py` has `tests/test_<script>.py`; LLMs are replaced by `tests/fakes.py`; `-m live` tests need a real server.
- The LLM server (vLLM, `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`, max length 16384, 32 parallel) may run on another machine: pass `--base-url`. Ask the user for the address if it is not known.
- `uv.lock` is in `.gitignore`. Optional dependency group `report` (matplotlib) is only for charts.
- Write repo files in English; talk to the user in Polish.
- Diagrams are welcome in docs: use Mermaid blocks inside the .md files (flows, pipelines, structure; the user's viewer renders them). Data charts stay matplotlib PNGs made by the scripts.

## Data
- `data/data.d00N.toml`: dialogues with a `single_choice_survey`. **`d004` is a story (`type = "Story"`) and is skipped on purpose** by the compress scripts; story compression may come later. Do not report or "fix" that.
- `data/compress.<method>.toml`: compressed versions (`target_words` = what was asked; `text_words`/`text_tokens` = what came out).
- `data/benchmark.<run>.<data_id>.toml`: benchmark results; `data/benchmark_replies.d001.json`: raw replies.

## Benchmark design (`utils/benchmark_compressed.py`) and why
- The reader gets **only the compressed text, as chat history**, then each question in a **separate chat** (questions would leak facts to each other). Temperature 0, 8 answer tokens, thinking off.
- **Order of answers drawn per query** from one seeded generator (default seed 42): repeatable, and no question is lucky every time.
- A missing or unclear letter counts as wrong and is also counted as `no_answer`.
- **Reference points in every run:** `No Text` (floor: what can be answered from the question alone) and `Original Text` (ceiling). The user found the floor confusing; explain it plainly when it comes up.
- The benchmark always uses the **whole** compressed text; never cut it, never use `target_words` as its length.
- **`--tag-legend`** (for DSPM): the DSPM author's legend of the tags `[CON] [DEC] [CODE] [EQ] [ENT] [STR]` before the text and "treat [CON] as hard rules" after it; the text stays in the chat history; reference points get no legend. The report shows DSPM as `DSPM-NoLegend` and `DSPM+Legend`.

## Report (`docs/benchmark.md`)
- To rerun everything (all runs, replies, report, notes check): the project skill `/benchmark-rerun` (`.claude/skills/benchmark-rerun/SKILL.md`).
- Generated: `uv run --group report utils/benchmark_report.py --notes docs/benchmark_notes.md --replies data/benchmark_replies.d001.json --output docs/benchmark.md`. Raw replies come from `utils/benchmark_replies.py` (repeats the benchmark queries exactly).
- **Every number is computed from the files.** Conclusions go only into the hand-written `docs/benchmark_notes.md` (copied verbatim to the top; it starts with the user's "Preliminary Results" warning, keep it). After a rerun, re-check every number in the notes.
- **Compare prompts p01/p02/p03 at the same real length** (fitted line per prompt), never by plain means over targets: the prompts write texts of different length, and longer texts keep more facts.
- Charts follow the dataviz rules: fixed color per method, marker shape + direct label as second encoding.

## Findings so far (benchmark of 2026-11-17, reader Gemma 26B)
- The reader refuses ("The provided text does not contain...") instead of guessing; about half the queries have no answer. Among answered questions it is mostly right.
- Frontier models keep the most facts per token (d001 only); `compress_Gemma26B` with thinking medium ignores the target length (texts often longer than the original); DSPM and SkLLM compress hardest and keep the fewest facts. DSPM is made for two speakers, the dialogues have three; with a small budget DSPM cuts entries to a few words. The tag legend changes DSPM's score only within noise.
- At the same length p01 is the weakest prompt; p02 and p03 are close.
- Noise: the same query repeated may differ; 1-2 questions between versions mean nothing.

## Open points (2026-11-17)
- `parse_letter` reads the refusal `None of the provided options (A,` as answer A (13 of 9300 replies; scores unchanged, `no_answer` slightly too low). Not fixed yet.
- Survey cue: in the d001 survey the correct answer is the longest of the five in 15 of 50 questions; all 6 questions the reader answered without any text were such questions. Not yet checked for d002/d003 or reported.
