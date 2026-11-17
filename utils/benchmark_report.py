#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# About: A utility script that turns the results of utils/benchmark_compressed.py (data/benchmark.*.toml) and,
#        optionally, the raw replies from utils/benchmark_replies.py into one Markdown report with charts.
# How to use it:
#   uv run --group report utils/benchmark_report.py --notes docs/benchmark_notes.md --replies data/benchmark_replies.d001.json --output docs/benchmark.md
#   uv run utils/benchmark_report.py --output docs/benchmark.md --no-charts      # without matplotlib
#
# The idea in short:
#   - Every number in the report is computed here from the result files; nothing is typed in by hand. Running the
#     script again after a new benchmark gives a report that matches the new files.
#   - Conclusions are a different matter: they need a human (or Claude) to look at the numbers. They live in a separate,
#     hand-written Markdown file (--notes), which is copied verbatim to the top of the report. So a regenerated report
#     keeps the notes, and the notes are the only place where an opinion can go stale.
#   - Charts are PNG files in --img-dir, linked from the report. matplotlib is needed only for them (the 'report'
#     dependency group); it is imported only when charts are drawn, so the rest works and is tested without it.
#
# Main decisions:
#   - Length is always the REAL length of the compressed text (its tokens), never the requested target_words.
#   - Prompts p01/p02/p03 are compared at the same real length (a fitted line per prompt, read at a length that all
#     three prompts produced), because a prompt that writes longer texts keeps more facts for that reason alone.
#   - The output report is overwritten on purpose (unlike the benchmark results): it is generated, not a source.
import sys
import re
import glob
import json
import math
import argparse
import statistics
import tomllib
from pathlib import Path
from collections import defaultdict, Counter
import benchmark_compressed as bc

PROMPTS        = ("p01", "p02", "p03")
METHOD_ORDER   = ["Gemma26B direct", "DSPM-NoLegend", "DSPM+Legend", "SkLLM", "Frontier models"]
SOURCE_ORDER   = ["Gemma26B_ThinkOff", "Gemma26B_ThinkMedium", "ChatGPT default", "Gemini 3 Medium", "Sonnet 5.5 Medium"]
LENGTH_BINS    = [0.04, 0.08, 0.15, 0.25, 0.4, 0.65, 1.0, 1.6, 3.0] # bin edges of the charts, share of the original length
MIN_PER_BIN    = 3    # a bin of the charts needs this many versions to get a median
REFERENCE      = ("No Text", "Original Text")
REFUSAL        = re.compile(r"^(The provided text|The text provided|None of the|I cannot|I am sorry|Please provide|Based on the text|The text does not)", re.I)
LETTER_CHOICE  = re.compile(r"(?:answer is|correct answer is|choose|select)[^A-E]{0,30}\b([A-E])\b|\*\*([A-E])\*\*|^\s*([A-E])\)?\s*$", re.M | re.I)

# ---------------------------------------------------------------------------------------------------- loading ----------

def parse_arguments() -> argparse.Namespace:
    """parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Writes a Markdown report (with charts) from the benchmark result files.")
    parser.add_argument("--results", type=str, default="data/benchmark.*.toml", help="Glob of benchmark result files (default: 'data/benchmark.*.toml')")
    parser.add_argument("--replies", type=str, default=None, help="Optional JSON file from benchmark_replies.py (adds the prompt and reply examples)")
    parser.add_argument("--notes",   type=str, default=None, help="Optional hand-written Markdown file copied verbatim to the top of the report")
    parser.add_argument("--output",  type=str, required=True, help="Path of the Markdown report to write (it is overwritten)")
    parser.add_argument("--img-dir", type=str, default=None, help="Folder for the charts (default: 'img' next to the report)")
    parser.add_argument("--no-charts", action="store_true", help="Do not draw charts (no matplotlib needed); the report then has no images")
    return parser.parse_args()

def run_name(compressed_path: str) -> str:
    """'data/compress.Frontier.p02.d001.toml' -> 'Frontier.p02.d001'"""
    stem = Path(compressed_path).stem
    return stem[len("compress."):] if stem.startswith("compress.") else stem

def method_of(compressed_path: str, tag_legend: bool = False) -> str:
    """the compression method of a compressed file, from its name; DSPM read with and without the tag legend are two systems"""
    name = run_name(compressed_path)
    if name.startswith("SkLLM"):    return "SkLLM"
    if name.startswith("DSPM"):     return "DSPM+Legend" if tag_legend or name.endswith("+Legend") else "DSPM-NoLegend"
    if name.startswith("Frontier"): return "Frontier models"
    return "Gemma26B direct"

def load_runs(paths: list) -> list:
    """loads benchmark result files: [{file, run, method, benchmark, config, results}], sorted by file name"""
    runs = list()
    for path in sorted(paths):
        with open(path, "rb") as f:
            t = tomllib.load(f)
        if t.get("benchmark", {}).get("type") != "Benchmark Compressed Dialogue":
            continue
        compressed, legend = t["benchmark"]["compressed_path"], t["benchmark"].get("tag_legend", False)
        runs.append({"file": Path(path).name, "run": run_name(compressed) + ("+Legend" if legend else ""), "method": method_of(compressed, legend),
                     "benchmark": t["benchmark"], "config": next(iter(t.get("configs", {}).values()), {}), "results": t["results"]})
    return runs

# ---------------------------------------------------------------------------------------------------- numbers ----------

def lsq_fit(points: list) -> tuple:
    """least-squares line of y on ln(x): returns (a, b, r) for y = a + b * ln(x); r is the correlation (nan if y is flat)"""
    xs = [math.log(x) for x, _ in points]; ys = [y for _, y in points]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs); sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys)); syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0:
        raise ValueError("all points have the same length, a line cannot be fitted")
    b = sxy / sxx
    return my - b * mx, b, (sxy / math.sqrt(sxx * syy) if syy else float("nan"))

def fit_value(fit: tuple, x: float) -> float:
    """value of a fitted line at length x, clipped to 0..1 (accuracy cannot leave that range)"""
    return min(1.0, max(0.0, fit[0] + fit[1] * math.log(x)))

def common_range(groups: dict):
    """the range of lengths that every group covers: (low, high), or None when they do not overlap"""
    low  = max(min(x for x, _ in pts) for pts in groups.values())
    high = min(max(x for x, _ in pts) for pts in groups.values())
    return (low, high) if low < high else None

def binned_median(points: list, edges=LENGTH_BINS, min_n=MIN_PER_BIN) -> list:
    """medians of y in length bins: [(bin center on the log scale, median)], only bins with at least min_n points"""
    out = list()
    for low, high in zip(edges[:-1], edges[1:]):
        ys = [y for x, y in points if low <= x < high and not math.isnan(y)]
        if len(ys) >= min_n:
            out.append((math.sqrt(low * high), statistics.median(ys)))
    return out

def answered_accuracy(x: dict) -> float:
    """correct / answered; nan when nothing was answered"""
    answered = x["questions"] - x["no_answer"]
    return x["correct"] / answered if answered else float("nan")

def per_dialogue(runs: list) -> dict:
    """{data_id: {"floor": No Text result, "ceiling": Original Text result, "versions": [(run, result)]}}

    The reference points are computed again in every run of a dialogue and can differ by a question (noise).
    Decision: the value that most runs got is used (on a tie, the first of them), so the report does not depend
    on which result file happens to come first.
    """
    out = defaultdict(lambda: {"floor": None, "ceiling": None, "versions": []})
    refs = defaultdict(list)
    for run in runs:
        for x in run["results"]:
            if x["type"] in REFERENCE:
                refs[(x["data_id"], x["type"])].append(x)
            else:
                out[x["data_id"]]["versions"].append((run, x))
    for (data_id, kind), xs in refs.items():
        votes = Counter((x["correct"], x["no_answer"]) for x in xs)
        top = max(votes.values())
        out[data_id]["floor" if kind == "No Text" else "ceiling"] = next(x for x in xs if votes[(x["correct"], x["no_answer"])] == top)
    return dict(sorted(out.items()))

def reference_differences(runs: list) -> list:
    """the reference points are computed again in every run of a dialogue; returns the ones that differ between runs"""
    seen = defaultdict(set)
    for run in runs:
        for x in run["results"]:
            if x["type"] in REFERENCE:
                seen[(x["data_id"], x["type"])].add((x["correct"], x["no_answer"]))
    return [(k, sorted(v)) for k, v in sorted(seen.items()) if len(v) > 1]

def prompt_table(runs: list, dialogues: dict) -> dict:
    """{(source_llm, data_id): {prompt: [result, ...]}} for the compressors that were run with every prompt"""
    rows = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for x in run["results"]:
            if x["prompt_id"] in PROMPTS:
                rows[(x["source_llm"], x["data_id"])][x["prompt_id"]].append(x)
    keep = {k: v for k, v in rows.items() if all(v.get(p) for p in PROMPTS)}
    order = lambda k: (SOURCE_ORDER.index(k[0]) if k[0] in SOURCE_ORDER else len(SOURCE_ORDER), k[0], k[1])
    return {k: keep[k] for k in sorted(keep, key=order)}

def prompt_same_length(table: dict, dialogues: dict) -> dict:
    """for every compressor: the fitted accuracy of each prompt at one common real length

    Returns {key: {"range": (low, high) or None, "at": x or None, "acc": {prompt: value}, "r": {prompt: r}}}.
    The common length is the middle (on the log scale) of the range that all three prompts really produced.
    """
    out = dict()
    for key, by_prompt in table.items():
        orig = dialogues[key[1]]["ceiling"]["text_tokens"]
        points = {p: [(x["text_tokens"] / orig, x["accuracy"]) for x in by_prompt[p]] for p in PROMPTS}
        fits = {p: lsq_fit(points[p]) for p in PROMPTS}
        rng = common_range(points)
        at = math.sqrt(rng[0] * rng[1]) if rng else None
        out[key] = {"range": rng, "at": at, "r": {p: fits[p][2] for p in PROMPTS},
                    "acc": {p: fit_value(fits[p], at) for p in PROMPTS} if at else {}}
    return out

def reply_form(row: dict) -> str:
    """'bare letter', 'letter in a longer reply', 'no letter' or 'failed'"""
    if row["finish"] == "error":
        return "failed"
    if row["parsed"] is None:
        return "no letter"
    return "bare letter" if re.fullmatch(r"[A-J]", row["reply"].strip()) else "letter in a longer reply"

def letter_choices(text: str) -> list:
    """explicit letter choices in a long reply ("the answer is B", "**B**", a line with only "B")"""
    return [x for t in LETTER_CHOICE.findall(text) for x in t if x]

# ---------------------------------------------------------------------------------------------------- report -----------

def pct(v: float, digits: int = 1) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}%}"

def chat_text(messages: list) -> str:
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)

def build_report(runs: list, replies: dict = None, notes: str = None, img: str = None, data_dir: Path = Path("data")) -> str:
    """the whole report as Markdown; img is the folder of the charts relative to the report, or None for no charts"""
    L = list(); w = L.append
    dialogues = per_dialogue(runs)
    config, bench = runs[0]["config"], runs[0]["benchmark"]
    if notes:
        w(notes.rstrip()); w("")
    w("# Benchmark: compressed dialogues read by an LLM")
    w("")
    w(f"Date: {bench['date']} · Reader: `{config.get('MODEL_NAME', '?')}` (config `{bench.get('reader_llm', '?')}`) · Seed: {bench['seed']}")
    w("")
    w("The reader LLM gets **only the compressed text** (as chat history) and answers a single-choice survey (5 answers, one correct) written for the original dialogue. "
      "The share of correct answers is the score of the compressed text. Every question is asked in a separate chat, and the order of answers is drawn anew for every query "
      "from one seeded generator, so a run can be repeated exactly. See `utils/benchmark_compressed.py` for details. "
      "This file is generated by `utils/benchmark_report.py`; the notes above it are written by hand.")
    w("")

    # ---- how to read, setup
    floors = [d["floor"]["accuracy"] for d in dialogues.values() if d["floor"]]
    w("## How to read the numbers")
    w("")
    w("- **Accuracy** = correct answers / questions. A missing or unclear answer counts as wrong.")
    w("- **No answer** = the reply had no clear letter (see `parse_letter` in the benchmark). The examples at the end of this file show what such replies say.")
    w(f"- **No Text** (floor): the reader sees only the questions and answers. Here: {pct(min(floors), 0)}–{pct(max(floors), 0)}. A random guess would give 20%; a reader that refuses instead of guessing scores lower.")
    w("- **Original Text** (ceiling): the reader sees the full dialogue. A compressed text is good when it is close to this number.")
    w("- **target words** is only the length the compressor was asked for. The benchmark always uses the whole text that really came out; its real length is in **words** and **tokens**.")
    w("- **% of original**: real tokens of the compressed text / tokens of the original text.")
    w("")
    w("## Setup")
    w("")
    n_queries = sum(x["questions"] for r in runs for x in r["results"]); n_noans = sum(x["no_answer"] for r in runs for x in r["results"])
    w(f"- Reader settings: temperature {config.get('TEMPERATURE')}, top_p {config.get('TOP_P')}, max answer tokens {config.get('MAX_ANSWER_TOKENS')}, reasoning {'on' if config.get('ENABLE_REASONING') else 'off'}, concurrency {config.get('CONCURRENCY')}")
    w(f"- {len(runs)} runs, {n_queries} LLM queries, {n_noans} of them without a clear answer ({pct(n_noans / n_queries)})")
    w("")
    w("Each run (raw results in `data/`):")
    w("")
    w("```")
    for r in runs:
        b = r["benchmark"]
        w(f"uv run utils/benchmark_compressed.py {b['data_path']} {b['compressed_path']}{' --tag-legend' if b.get('tag_legend') else ''} --base-url <LLM server> --output data/{r['file']}")
    w("```")
    w("")

    # ---- charts
    if img:
        w("## Charts: how each method works depending on the length of the compressed text")
        w("")
        w("Each dot is one compressed version, the line is the median of the versions in a length bin (bins with fewer than "
          f"{MIN_PER_BIN} versions are not drawn). The x axis is the real length as a share of the original, on a log scale; right of the dashed line the text is longer than the original.")
        w("")
        w(f"![Accuracy against the length of the compressed text, per method and dialogue]({img}/benchmark_accuracy_vs_length.png)")
        w("")
        w(f"![Share of questions without an answer against the length of the compressed text]({img}/benchmark_noanswer_vs_length.png)")
        w("")
        w(f"![Accuracy among the answered questions against the length of the compressed text]({img}/benchmark_answered_accuracy_vs_length.png)")
        w("")

    # ---- per dialogue
    for data_id, d in dialogues.items():
        ot, nt = d["ceiling"], d["floor"]
        if not ot or not nt:
            continue
        orig = ot["text_tokens"]
        w(f"## {data_id}: {ot['questions']} questions, original text {ot['text_words']} words / {orig} tokens")
        w("")
        w(f"Floor (no text): **{nt['correct']}/{nt['questions']} = {pct(nt['accuracy'])}** ({nt['no_answer']} no answer). "
          f"Ceiling (original text): **{ot['correct']}/{ot['questions']} = {pct(ot['accuracy'])}** ({ot['no_answer']} no answer).")
        w("")
        groups = defaultdict(list)
        for run, x in d["versions"]:
            groups[run["method"]].append(x)
        for title, only_shorter in (("Summary by method", False), ("Only versions that are really shorter than the original", True)):
            w(f"### {title}")
            w("")
            w("| method | versions | % of original (median) | accuracy mean | accuracy best | no answer (mean) | accuracy among answered (mean) |")
            w("|---|---:|---:|---:|---:|---:|---:|")
            for m in METHOD_ORDER:
                xs = [x for x in groups.get(m, []) if not only_shorter or x["text_tokens"] < orig]
                if not xs:
                    continue
                answered = [answered_accuracy(x) for x in xs if not math.isnan(answered_accuracy(x))]
                w(f"| {m} | {len(xs)}{f' / {len(groups[m])}' if only_shorter else ''} | {statistics.median(x['text_tokens'] for x in xs) / orig:.0%} | "
                  f"{pct(statistics.mean(x['accuracy'] for x in xs))} | {pct(max(x['accuracy'] for x in xs))} | "
                  f"{statistics.mean(x['no_answer'] / x['questions'] for x in xs):.0%} | {pct(statistics.mean(answered)) if answered else '-'} |")
            w("")
        for m in METHOD_ORDER:
            xs = groups.get(m)
            if not xs:
                continue
            w(f"<details><summary>{data_id}: every version of {m} ({len(xs)})</summary>")
            w("")
            w("| prompt | source LLM | target words | words | tokens | % of original | correct | accuracy | no answer |")
            w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
            for x in sorted(xs, key=lambda x: (x["source_llm"], x["prompt_id"], -x["target_words"])):
                w(f"| {x['prompt_id']} | {x['source_llm']} | {x['target_words']} | {x['text_words']} | {x['text_tokens']} | {x['text_tokens'] / orig:.0%} | "
                  f"{x['correct']}/{x['questions']} | {pct(x['accuracy'])} | {x['no_answer']} |")
            w("")
            w("</details>")
            w("")

    # ---- prompts
    table = prompt_table(runs, dialogues)
    if table:
        label = lambda k: {"Gemma26B_ThinkOff": "Gemma26B, thinking off", "Gemma26B_ThinkMedium": "Gemma26B, thinking medium"}.get(k[0], k[0])
        same = prompt_same_length(table, dialogues)
        w("## Prompts p01, p02, p03 compared on the same compressor")
        w("")
        w("A longer text keeps more facts, and the prompts produce texts of a different real length from the same target. "
          "So the prompts are compared **at the same real length**: one line per prompt (least squares of accuracy on log(length)), read at a length that all three prompts produced.")
        w("")
        prompts_file = data_dir / "prompts.compress.toml"
        if prompts_file.is_file():
            w("<details><summary>The three prompts, in full (from <code>data/prompts.compress.toml</code>)</summary>")
            w("")
            for pr in tomllib.load(open(prompts_file, "rb")).get("prompts", []):
                w(f"**{pr['prompt_id']}**"); w(""); w("```text")
                if pr.get("system_prompt_template", "").strip():
                    w("[system]"); w(pr["system_prompt_template"].strip()); w("")
                w("[user]"); w(pr["user_prompt_template"].strip()); w("```"); w("")
            w("</details>")
            w("")
        if img:
            w(f"![Accuracy against real length, one line per prompt, compressors run with every prompt]({img}/benchmark_prompts.png)")
            w("")
            w("Compare the lines at the same x, best inside the gray range (the lengths all three prompts produced). Each line is drawn only over the lengths its prompt really produced.")
            w("")
        w("### Accuracy at the same length")
        w("")
        w("\"fit r\" tells how well the length explains the accuracy of a prompt (1.0 = all versions on the line); with a low r the comparison means little. "
          "Each line rests on a few versions only, so differences of a few points are not reliable.")
        w("")
        w("| compressor | dialogue | versions per prompt | common length range | compared at | p01 | p02 | p03 | fit r p01 / p02 / p03 |")
        w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for key, s in same.items():
            n = "/".join(str(len(table[key][p])) for p in PROMPTS)
            rs = " / ".join(f"{s['r'][p]:.2f}" for p in PROMPTS)
            if not s["at"]:
                w(f"| {label(key)} | {key[1]} | {n} | none | - | - | - | - | {rs} |"); continue
            best = max(s["acc"].values())
            cells = " | ".join(f"**{s['acc'][p]:.0%}**" if s["acc"][p] == best else f"{s['acc'][p]:.0%}" for p in PROMPTS)
            w(f"| {label(key)} | {key[1]} | {n} | {s['range'][0]:.0%}–{s['range'][1]:.0%} | {s['at']:.0%} | {cells} | {rs} |")
        w("")
        w("### Real length of the texts")
        w("")
        w("\"words / target\" is the median of real words / target words (1.0 = the requested length was kept). \"not shorter\" counts versions as long as the original or longer.")
        w("")
        w("| compressor | dialogue | words / target p01 / p02 / p03 | % of original tokens (median) p01 / p02 / p03 | not shorter p01 / p02 / p03 |")
        w("|---|---|---:|---:|---:|")
        for key, by_prompt in table.items():
            orig = dialogues[key[1]]["ceiling"]["text_tokens"]
            ratio = " / ".join(f"{statistics.median(x['text_words'] / x['target_words'] for x in by_prompt[p] if x['target_words']):.2f}" for p in PROMPTS)
            share = " / ".join(f"{statistics.median(x['text_tokens'] / orig for x in by_prompt[p]):.0%}" for p in PROMPTS)
            longer = " / ".join(str(sum(x["text_tokens"] >= orig for x in by_prompt[p])) for p in PROMPTS)
            w(f"| {label(key)} | {key[1]} | {ratio} | {share} | {longer} |")
        w("")
        w("<details><summary>Every version, side by side for the three prompts</summary>")
        w("")
        for key, by_prompt in table.items():
            w(f"**{label(key)}, {key[1]}**")
            w("")
            w("| target words | p01 words | p01 accuracy | p02 words | p02 accuracy | p03 words | p03 accuracy |")
            w("|---:|---:|---:|---:|---:|---:|---:|")
            by_target = {p: {x["target_words"]: x for x in by_prompt[p]} for p in PROMPTS}
            for t in sorted(set.intersection(*(set(by_target[p]) for p in PROMPTS)), reverse=True):
                best = max(by_target[p][t]["accuracy"] for p in PROMPTS)
                cells = []
                for p in PROMPTS:
                    x = by_target[p][t]; a = f"{x['accuracy']:.0%}"
                    cells += [str(x["text_words"]), f"**{a}**" if x["accuracy"] == best else a]
                w(f"| {t} | " + " | ".join(cells) + " |")
            w("")
        w("</details>")
        w("")

    # ---- prompts and replies of the reader
    if replies:
        w(replies_section(replies, runs))

    # ---- repeatability
    w("## Repeatability check")
    w("")
    diffs = reference_differences(runs)
    if diffs:
        w("The two reference points are computed again in every run of the same dialogue. These came out different between runs, as (correct, no answer):")
        w("")
        for (data_id, kind), values in diffs:
            w(f"- {data_id} {kind}: {values}")
        w("")
        w("So a difference of this size between two versions is noise, not a real difference.")
    else:
        w("The two reference points are computed again in every run of the same dialogue, and they came out identical in all runs.")
    w("")
    return "\n".join(L) + "\n"

def replies_section(replies: dict, runs: list) -> str:
    """the examples: real prompts, real replies, how the parser read them"""
    L = list(); w = L.append
    rows, long_rows = replies["rows"], replies.get("long", [])
    n = len(rows)
    forms = Counter(reply_form(r) for r in rows)
    data_toml = bc.load_toml(Path(replies["data_path"]))
    surveys, _ = bc.load_surveys(data_toml)

    # the benchmark scores for the same runs, to show that the replies belong to them
    bench = {r["run"]: r["results"] for r in runs if r["benchmark"]["data_path"] == replies["data_path"]}
    mine = Counter((r["run"], r["vidx"]) for r in rows if r["parsed"] == r["expected"])
    texts = {(r["run"], r["vidx"]) for r in rows}
    same = [abs(bench[run][vidx]["correct"] - mine[(run, vidx)]) for run, vidx in texts if run in bench and vidx < len(bench[run])]

    w("## Examples: real prompts")
    w("")
    example = next((r for r in rows if r["type"] not in REFERENCE and r["qidx"] == 0), rows[0])
    q = {"question": example["question"], "answers": example["answers"], "correct_letter": example["expected"]}
    w(f"Every question is a separate chat. Below is the real text sent to the reader for question 1 about one version "
      f"({example['source_llm']}, prompt {example['prompt_id']}, target {example['target_words']} words, run {example['run']}); "
      f"the compressed dialogue is replaced by the tag `<COMPRESSED_DIALOGUE>`. The correct answer in this query is **{example['expected']}**.")
    w("")
    w("```text"); w(chat_text(bc.build_messages("<COMPRESSED_DIALOGUE>", q, example.get("tag_legend", False)))); w("```")
    w("")
    nt = next((r for r in rows if r["type"] == "No Text" and r["qidx"] == 0 and r["run"] == example["run"]), None)
    if nt:
        qn = {"question": nt["question"], "answers": nt["answers"], "correct_letter": nt["expected"]}
        w("**Why there is a prompt with no dialogue at all (No Text).** It is a reference point, not a compression method: it measures how many questions can be "
          "answered from the question and its answers alone (general knowledge, an answer that gives itself away, guessing). A compressed text helps only as far as it scores above it. "
          f"The reader then sees only this (the same question with its own order of answers; correct answer **{nt['expected']}**):")
        w("")
        w("```text"); w(chat_text(bc.build_messages("", qn))); w("```")
        w("")
    w("### What goes into `<COMPRESSED_DIALOGUE>`")
    w("")
    w("**The whole compressed text, exactly as it is stored in the compressed file**; nothing is cut, and `target_words` is not used.")
    tok = [r for r in rows if r.get("prompt_tokens") and r["type"] not in REFERENCE and r["finish"] != "error"]
    if tok:
        spread = defaultdict(list)
        for r in tok:
            spread[r["qidx"]].append(r["prompt_tokens"] - r["text_tokens"])
        worst = max(max(v) - min(v) for v in spread.values())
        same_q = "is exactly the same for every version" if worst == 0 else f"varies by at most {worst} tokens"
        w(f"Check from the LLM server's own count: for the same question, *prompt tokens − tokens of the stored text* {same_q} "
          f"({len(tok)} queries), while the texts are {min(r['text_tokens'] for r in tok)}–{max(r['text_tokens'] for r in tok)} tokens long. "
          "A cut text would make this difference jump by thousands of tokens for the long versions.")
    w("")
    w("One real version per method, in full (the shortest of each):")
    w("")
    shown = set()
    for path in replies["compressed_paths"]:
        method = method_of(path, replies.get("tag_legend", False))
        if method in shown:
            continue
        results = bc.load_toml(Path(path)).get("results", [])
        cands = [r for r in results if r.get("data_id") in surveys and r.get("text")]
        if not cands:
            continue
        r = min(cands, key=lambda r: r.get("text_tokens", 10 ** 9)); shown.add(method)
        w(f"<details><summary>{method}: {r.get('source_llm', '-')}, prompt {r.get('prompt_id', '-')}, target {r.get('target_words', 0)} words — real: {r.get('text_words', '?')} words, {r.get('text_tokens', '?')} tokens</summary>")
        w(""); w("```text"); w(r["text"].strip()); w("```"); w(""); w("</details>"); w("")

    w("## Examples: real replies of the reader")
    w("")
    w(f"`utils/benchmark_replies.py` asked all {n} queries of the runs {', '.join(sorted({r['run'] for r in rows}))} once more, exactly as the benchmark asks them "
      f"(same prompts, settings, seed {replies['seed']} and order of answers), and kept the raw replies. "
      + (f"The scores of this repeat match the benchmark files: {len(same) - sum(1 for s in same if s)} of {len(same)} texts got the same number of correct answers, "
         f"the others differ by at most {max(same) if same else 0}." if same else ""))
    w("")
    w("| form of the reply | queries | share |")
    w("|---|---:|---:|")
    for form in ("bare letter", "letter in a longer reply", "no letter", "failed"):
        if forms[form] or form != "failed":
            w(f"| {form} | {forms[form]} | {pct(forms[form] / n)} |")
    w("")
    by = defaultdict(Counter)
    for r in rows:
        by["reference points" if r["type"] in REFERENCE else method_of(r["run"])][reply_form(r)] += 1
    w("| source | queries | bare letter | letter in a longer reply | no letter |")
    w("|---|---:|---:|---:|---:|")
    for name in ["reference points", *METHOD_ORDER]:
        c = by.get(name)
        if c:
            t = sum(c.values())
            w(f"| {name} | {t} | {c['bare letter'] / t:.0%} | {c['letter in a longer reply'] / t:.0%} | {c['no letter'] / t:.0%} |")
    w("")
    w("### When the reader does not answer by the schema")
    w("")
    no_letter = [r for r in rows if reply_form(r) == "no letter"]
    beginnings = Counter(" ".join(r["reply"].split()[:4]) for r in no_letter)
    w(f"{len(no_letter)} replies had no clear letter. The replies are cut at {replies['config'].get('MAX_ANSWER_TOKENS')} tokens (the benchmark setting); their most common beginnings:")
    w("")
    w("| beginning of the reply | queries | share of no-letter replies |")
    w("|---|---:|---:|")
    top = beginnings.most_common(8)
    for k, v in top:
        ex = next(r["reply"] for r in no_letter if " ".join(r["reply"].split()[:4]) == k)
        w(f"| `{ex}` | {v} | {pct(v / len(no_letter))} |")
    if no_letter:
        rest = len(no_letter) - sum(v for _, v in top)
        w(f"| (other beginnings) | {rest} | {pct(rest / len(no_letter))} |")
    w("")
    if long_rows:
        w(f"Some of them asked again with up to {replies['long_tokens']} tokens (each counts as not answered in the benchmark):")
        w("")
        seen = set()
        for r in long_rows:
            key = " ".join(r["reply"].split()[:3])
            if key in seen or r["long_finish"] != "stop" or len(r["long_reply"]) > 520:
                continue
            seen.add(key)
            who = "reference point (No Text)" if r["type"] == "No Text" else f"{r['run']}, {r['source_llm']}"
            w(f"- **{who}**, question {r['qidx'] + 1} (correct: {r['expected']}), short reply `{r['reply']}`")
            w("  ")
            w("  > " + " ".join(r["long_reply"].split()))
            w("")
            if len(seen) >= 7:
                break
        committed = [r for r in long_rows if letter_choices(r["long_reply"])]
        ok = sum(1 for r in committed if letter_choices(r["long_reply"])[-1].upper() == r["expected"])
        stopped = sum(1 for r in long_rows if r["long_finish"] == "stop")
        w("### Does the short answer limit hide real answers?")
        w("")
        w(f"A random sample of {len(long_rows)} of the {len(no_letter)} no-letter replies was asked again with {replies['long_tokens']} tokens: {stopped} ended on their own. "
          f"**{len(committed)}** of them ({pct(len(committed) / len(long_rows), 0)}) contain an explicit letter choice (pattern: \"answer is X\", \"**X**\" or a line with only the letter), "
          f"and **{ok}** of those are correct. The numbers come from a sample and a simple pattern, so read them as an order of magnitude.")
        w("")
    w("### How the parser read replies outside the schema")
    w("")
    longer = [r for r in rows if reply_form(r) == "letter in a longer reply"]
    refusals = [r for r in longer if REFUSAL.match(r["reply"].strip())]
    w(f"{len(longer)} replies were not a bare letter but still got a letter from `parse_letter`; {len(refusals)} of them look like refusals "
      "(\"The provided text does not...\"). All distinct ones:")
    w("")
    if longer:
        w("| reply | times | letter taken | counted as correct |")
        w("|---|---:|---|---:|")
        groups = defaultdict(list)
        for r in longer:
            groups[(r["reply"], r["parsed"])].append(r)
        for (reply, letter), rs in sorted(groups.items(), key=lambda g: -len(g[1])):
            w(f"| `{reply}` | {len(rs)} | {letter} | {sum(1 for r in rs if r['parsed'] == r['expected'])} of {len(rs)} |")
        w("")
    return "\n".join(L) + "\n"

# ---------------------------------------------------------------------------------------------------- charts -----------

def draw_charts(runs: list, img_dir: Path) -> list:
    """draws the PNG charts linked from the report, returns their paths (needs matplotlib: uv run --group report ...)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    # colors: categorical slots in a fixed order, checked for color-blind separation; a marker shape per series as well
    SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    # Color follows the method: the four older methods keep their colors; DSPM+Legend gets violet, which passes the
    # adjacent-pair color checks in this order (magenta and red were too close to the orange of DSPM-NoLegend).
    # With five series some pairs (blue/violet, orange/yellow) are close, so every method also has its own marker
    # shape and a direct label at the end of its line.
    METHOD_STYLE = {"Gemma26B direct": ("#2a78d6", "o"), "DSPM-NoLegend": ("#eb6834", "s"), "DSPM+Legend": ("#6250d6", "v"), "SkLLM": ("#1baf7a", "^"), "Frontier models": ("#eda100", "D")}
    PROMPT_STYLE = {"p01": ("#2a78d6", "o"), "p02": ("#eb6834", "s"), "p03": ("#1baf7a", "^")}
    img_dir.mkdir(parents=True, exist_ok=True)
    dialogues = per_dialogue(runs)
    ids = [k for k, d in dialogues.items() if d["ceiling"] and d["versions"]]
    written = list()

    def style(ax):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors=MUTED, labelsize=9, length=3)
        ax.set_xscale("log"); ax.minorticks_off()

    def x_ticks(ax):
        lo, hi = ax.get_xlim()
        ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5) if lo <= t <= hi]
        ax.set_xticks(ticks); ax.set_xticklabels([f"{t:.0%}" for t in ticks])

    def y_axis(ax, low=0.0):
        ax.set_ylim(low - 0.02, 1.04)
        ticks = [t / 10 for t in range(int(low * 10), 11, 2 if low == 0 else 1)]
        ax.set_yticks(ticks); ax.set_yticklabels([f"{t:.0%}" for t in ticks])

    def original_line(ax):
        if ax.get_xlim()[0] < 1.0 < ax.get_xlim()[1]:
            ax.axvline(1.0, color=AXIS, lw=1.2, ls=(0, (4, 3)))
            ax.text(1.03, 0.02, "as long as\nthe original", color=MUTED, fontsize=7.5, va="bottom", ha="left", transform=ax.get_xaxis_transform(),
                    bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none", alpha=0.85))

    def label_line_ends(ax, ends):
        """names the lines at their right ends; labels that would overlap are moved apart vertically"""
        ax.figure.canvas.draw()
        placed = list()
        for m, (x, y) in sorted(ends, key=lambda e: -e[1][1]):
            px, py = ax.transData.transform((x, y))
            dy = 0
            while any(abs(px - qx) < 90 and abs(py + dy - qy) < 11 for qx, qy in placed):
                dy -= 11
            placed.append((px, py + dy))
            ax.annotate(m, (x, y), xytext=(6, dy * 72 / ax.figure.dpi), textcoords="offset points", fontsize=8, color=INK2, va="center", ha="left", zorder=6)

    def save(fig, name, title, subtitle, handles, height):
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9.5, labelcolor=INK2)
        fig.suptitle(title, x=0.012, y=0.995, ha="left", fontsize=13, color=INK, fontweight="bold")
        fig.text(0.012, 1 - 0.42 / height, subtitle, fontsize=9, color=INK2, ha="left", va="top")
        fig.tight_layout(rect=(0, 0.6 / height, 1, 1 - 0.62 / height))
        path = img_dir / name
        fig.savefig(path, facecolor=SURFACE); plt.close(fig); written.append(path)

    # 1-3: one metric against length, one panel per dialogue, one series per method
    metrics = [
        ("accuracy", "Accuracy of the reader vs length of the compressed text", "correct answers / questions", "benchmark_accuracy_vs_length.png", 0.0),
        ("no_answer", "How often the reader gives no answer vs length of the compressed text", "share of questions without an answer", "benchmark_noanswer_vs_length.png", 0.0),
        ("answered", "Accuracy among the questions the reader did answer vs length of the compressed text", "correct / answered", "benchmark_answered_accuracy_vs_length.png", 0.4),
    ]
    value = {"accuracy": lambda x: x["accuracy"], "no_answer": lambda x: x["no_answer"] / x["questions"], "answered": answered_accuracy}
    for metric, title, ylabel, name, low in metrics:
        height = 5.6
        fig, axes = plt.subplots(1, len(ids), figsize=(15.5, height), dpi=110, squeeze=False, sharey=True)
        fig.patch.set_facecolor(SURFACE)
        hidden = 0
        for ax, did in zip(axes[0], ids):
            d = dialogues[did]; orig = d["ceiling"]["text_tokens"]; style(ax); y_axis(ax, low)
            xs_all = [x["text_tokens"] / orig for _, x in d["versions"]]
            ax.set_xlim(min(xs_all) / 1.25, max(xs_all) * 1.25)
            if metric == "accuracy":
                ax.axhline(d["floor"]["accuracy"], color=AXIS, lw=1.0, ls=(0, (1, 2)))
                ax.text(0.01, d["floor"]["accuracy"] + 0.01, f"no text = {d['floor']['accuracy']:.0%}", color=MUTED, fontsize=8, va="bottom", ha="left", transform=ax.get_yaxis_transform(),
                        bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none", alpha=0.85), zorder=7)
                ax.axhline(d["ceiling"]["accuracy"], color=AXIS, lw=1.0)
            ends = list()
            for m in METHOD_ORDER:
                pts = [(x["text_tokens"] / orig, value[metric](x)) for run, x in d["versions"] if run["method"] == m]
                pts = [(a, b) for a, b in pts if not math.isnan(b)]
                if not pts: continue
                hidden += sum(1 for _, b in pts if b < low)
                color, marker = METHOD_STYLE[m]
                ax.scatter([a for a, _ in pts], [b for _, b in pts], s=34, marker=marker, color=color, alpha=0.5, edgecolors=SURFACE, linewidths=0.8, zorder=3)
                line = binned_median(pts)
                if len(line) >= 2:
                    ax.plot([a for a, _ in line], [b for _, b in line], color=color, lw=2.2, zorder=4)
                    ax.scatter([a for a, _ in line], [b for _, b in line], s=46, marker=marker, color=color, edgecolors=SURFACE, linewidths=1.4, zorder=5)
                    ends.append((m, line[-1]))
            label_line_ends(ax, ends)
            original_line(ax); x_ticks(ax)
            ax.set_title(f"{did}: {d['ceiling']['questions']} questions, original {orig} tokens", loc="left", fontsize=10.5, color=INK, pad=8)
            ax.set_xlabel("real length of the compressed text, % of the original (log scale)", fontsize=9, color=INK2)
        axes[0][0].set_ylabel(ylabel, fontsize=9.5, color=INK2)
        handles = [Line2D([0], [0], color=c, lw=2.2, marker=k, markersize=7, markeredgecolor=SURFACE, label=m) for m, (c, k) in METHOD_STYLE.items()]
        subtitle = f"Dots: one compressed version each. Lines: median per length bin (only bins with {MIN_PER_BIN} or more versions)."
        if low > 0:
            subtitle += f" The axis starts at {low:.0%}; {hidden} version(s) below it are not shown."
        save(fig, name, title, subtitle, handles, height)

    # 4: prompts, one panel per compressor and dialogue
    table = prompt_table(runs, dialogues)
    if table:
        keys = list(table)
        nc = 3; nr = math.ceil(len(keys) / nc); height = 4.3 * nr + 1.2
        fig, axes = plt.subplots(nr, nc, figsize=(15.5, height), dpi=110, squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        for i, key in enumerate(keys):
            ax = axes[i // nc][i % nc]; style(ax); y_axis(ax)
            orig = dialogues[key[1]]["ceiling"]["text_tokens"]
            points = {p: [(x["text_tokens"] / orig, x["accuracy"]) for x in table[key][p]] for p in PROMPTS}
            allx = [a for pts in points.values() for a, _ in pts]
            ax.set_xlim(min(allx) / 1.25, max(allx) * 1.25)
            rng = common_range(points)
            if rng:
                ax.axvspan(rng[0], rng[1], color=GRID, alpha=0.45, lw=0, zorder=0)
            for p in PROMPTS:
                color, marker = PROMPT_STYLE[p]; pts = points[p]
                ax.scatter([a for a, _ in pts], [b for _, b in pts], s=38, marker=marker, color=color, alpha=0.75, edgecolors=SURFACE, linewidths=0.9, zorder=4)
                fit = lsq_fit(pts); lo, hi = min(a for a, _ in pts), max(a for a, _ in pts)
                gx = [lo * (hi / lo) ** (k / 49) for k in range(50)]
                ax.plot(gx, [fit_value(fit, x) for x in gx], color=color, lw=2.2, zorder=3)
            original_line(ax); x_ticks(ax)
            ax.set_title(f"{key[0]} · {key[1]}", loc="left", fontsize=10, color=INK, pad=6)
            if i % nc == 0: ax.set_ylabel("correct answers / questions", fontsize=9, color=INK2)
            ax.set_xlabel("real length, % of the original (log scale)", fontsize=9, color=INK2)
        for j in range(len(keys), nr * nc):
            axes[j // nc][j % nc].axis("off")
        handles = [Line2D([0], [0], color=c, lw=2.2, marker=k, markersize=7, markeredgecolor=SURFACE, label=p) for p, (c, k) in PROMPT_STYLE.items()]
        handles.append(Patch(color=GRID, alpha=0.8, label="lengths produced by all three prompts"))
        save(fig, "benchmark_prompts.png", "Prompts p01 / p02 / p03: accuracy vs real length of the compressed text",
             "Dots: one compressed version each. Line: least squares of accuracy on log(length), drawn only over the lengths that prompt produced. Compare at the same x.",
             handles, height)
    return written

# ---------------------------------------------------------------------------------------------------- main -------------

def main() -> None:
    args = parse_arguments()
    paths = [p for p in glob.glob(args.results) if p.endswith(".toml")]
    runs = load_runs(paths)
    if not runs:
        print(f"Error: No benchmark result files found for: '{args.results}'", file=sys.stderr)
        sys.exit(1)
    replies = None
    if args.replies:
        with open(args.replies, encoding="utf-8") as f:
            replies = json.load(f)
    notes = Path(args.notes).read_text(encoding="utf-8") if args.notes else None
    output_path = Path(args.output)
    img_dir = Path(args.img_dir) if args.img_dir else output_path.parent / "img"
    img_rel = None
    if not args.no_charts:
        written = draw_charts(runs, img_dir)
        img_rel = Path(img_dir).resolve().relative_to(output_path.parent.resolve()).as_posix() if img_dir.resolve().is_relative_to(output_path.parent.resolve()) else str(img_dir)
        print(f"Charts: {', '.join(str(p) for p in written)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(runs, replies, notes, img_rel, data_dir=Path(runs[0]["benchmark"]["data_path"]).parent), encoding="utf-8")
    print(f"Report from {len(runs)} result file(s): '{output_path}'")

if __name__ == "__main__":
    main()
