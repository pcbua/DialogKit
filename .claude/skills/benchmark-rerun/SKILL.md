---
name: benchmark-rerun
description: Rerun the DialogKit compressed-dialogue benchmark end to end - all benchmark runs (incl. DSPM with and without the tag legend), the raw replies, the generated report docs/benchmark.md - and re-check every number in docs/benchmark_notes.md. Use after a new compression method or prompt, a change to the benchmark/report scripts, a new reader model, or when the user asks to "puścić benchmark od nowa" / update the report.
---

# Benchmark rerun

Read `CLAUDE.md` first (benchmark design and report rules). Every number that ends up in the repo must come from a
run made with the scripts of current `origin/main`.

## 1. Prepare
- `git fetch origin` and a new branch from `origin/main` (e.g. `results/benchmark-<what>`).
- Server address: ask the user, or take it from Claude's memory; pass it as `--base-url <URL>`.
  Check it answers: `curl -s -m 8 <URL>/v1/models` (expect model `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`).
  If it does not answer, ask the user to start it; do not run against `localhost` by mistake.
- `uv run pytest` passes.
- The scripts refuse to overwrite outputs: move the old result files away only after the new runs succeed
  (`git rm` them in the commit, or run into a temporary folder and move).

## 2. Benchmark runs (one per compressed file and data id)
`uv run utils/benchmark_compressed.py data/data.<dNNN>.toml data/compress.<X>.toml --base-url <URL> --output data/benchmark.<X>.<dNNN>.toml`
- `compress.Gemma26B`, `compress.DSPM`, `compress.SkLLM`: for d001, d002, d003.
- `compress.Frontier.p01/p02/p03.d001`: d001 only (output `data/benchmark.Frontier.pNN.d001.toml`, no double `.d001`).
- **DSPM again with the legend**: same command plus `--tag-legend`, output `data/benchmark.DSPM-Legend.<dNNN>.toml`.
- d004 is a story and has no compressions: skipped on purpose.
- Add any new `compress.*.toml` the same way. Default seed 42; do not change prompts or reader settings.
- Long: run in the background, a few at a time; check each finishes without errors.

## 3. Check the runs
- 15 files (or more with new methods); every run has both reference points.
- `No Text` and `Original Text` of the same data id should be almost equal across runs (temperature 0, same seed);
  small differences are server noise. The report takes the value most runs got.
- Many `no_answer` are normal (the reader refuses); a run with mostly errors/timeouts is not: diagnose and rerun.

## 4. Raw replies (d001)
`uv run utils/benchmark_replies.py data/data.d001.toml data/compress.Frontier.p0*.d001.toml data/compress.Gemma26B.toml data/compress.DSPM.toml data/compress.SkLLM.toml --base-url <URL> --output data/benchmark_replies.d001.json`

## 5. Report
`uv run --group report utils/benchmark_report.py --notes docs/benchmark_notes.md --replies data/benchmark_replies.d001.json --output docs/benchmark.md`
- Look at the four charts in `docs/img/benchmark_*.png` (labels must not overlap, axes readable).

## 6. Re-check the notes (do not skip)
`docs/benchmark_notes.md` is hand-written and copied verbatim to the top of the report. After a rerun, **check every
number and claim in it against the new report**, fix what changed, and regenerate the report. Keep the user's
"Preliminary Results" warning at the top. Compare prompts only at the same real length; never use `target_words`
as the length.

## 7. Deliver
- Commit results + report (+ notes) with a message saying what was rerun and why; open the PR yourself
  (`pr` skill or `CONTRIBUTING.md`), wait for CI, report to the user in Polish: main changes in the numbers,
  anything surprising, PR link.
- Update the "Findings" and "Open points" in `CLAUDE.md` if they changed.
