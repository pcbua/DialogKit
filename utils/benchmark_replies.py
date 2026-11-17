#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# About: A utility script that repeats the queries of utils/benchmark_compressed.py and keeps the raw replies of the LLM.
#        The benchmark itself saves only the scores; this script shows what the reader actually wrote, e.g. when it did
#        not answer with a single letter. The output (JSON) is read by utils/benchmark_report.py.
# How to use it:
#   uv run utils/benchmark_replies.py data/data.d001.toml data/compress.Frontier.p02.d001.toml --output data/benchmark_replies.d001.json
#   uv run utils/benchmark_replies.py data/data.d001.toml data/compress.*.toml --base-url http://192.168.0.53:8000 --output data/benchmark_replies.d001.json
#
# The idea in short:
#   - Every compressed file is treated as one separate benchmark run, exactly as if utils/benchmark_compressed.py was
#     started for it: the two reference points first, then the compressed versions in file order, and a new order
#     of answers for every query, drawn from one generator seeded with --seed. With the same seed the queries are the
#     same as in the benchmark, so the replies here belong to the scores in the benchmark files.
#   - The prompts, the LLM settings and the parser are taken from benchmark_compressed.py (imported, not copied),
#     so this script cannot drift away from the benchmark.
#   - Second pass: a random sample of the replies without a letter is asked again with a much larger answer budget,
#     to see whether the 8-token limit of the benchmark cuts off answers that would come after a long introduction.
import sys
import argparse
import asyncio
import json
import random
import tomllib
from datetime import date
from pathlib import Path
import benchmark_compressed as bc
from openai import AsyncOpenAI

DEFAULT_LONG_SAMPLE = 400 # letter-less replies asked again with a larger budget
DEFAULT_LONG_TOKENS = 200 # the larger budget, in tokens
QUERY_ATTEMPTS      = 3   # a failed query is tried again; a reply of a failed query is marked with finish = "error"

def parse_arguments() -> argparse.Namespace:
    """parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Repeats the queries of benchmark_compressed.py for one data file and one or more compressed files, and saves the raw replies of the LLM to a JSON file.")
    parser.add_argument("data_path",        type=str, help="Path to the data TOML file with the survey (e.g., 'data/data.d001.toml')")
    parser.add_argument("compressed_paths", type=str, nargs="+", help="Paths to TOML files with compressed dialogues; each is one benchmark run (e.g., 'data/compress.SkLLM.toml')")
    parser.add_argument("--output",         type=str, required=True, help="Path to the output JSON file (must not exist)")
    parser.add_argument("--seed",           type=int, default=bc.DEFAULT_SEED, help=f"Seed for the order of answers, the same as in the benchmark (default: {bc.DEFAULT_SEED})")
    parser.add_argument("--base-url",       type=str, default=None, help=f"Address of the LLM server (default: the BASE_URL of benchmark_compressed.py, now '{bc.LLM_CONFIG['BASE_URL']}')")
    parser.add_argument("--long-sample",    type=int, default=DEFAULT_LONG_SAMPLE, help=f"How many letter-less replies to ask again with a larger budget; 0 turns it off (default: {DEFAULT_LONG_SAMPLE})")
    parser.add_argument("--tag-legend",     action="store_true", help="Give the DSPM tag legend with the compressed texts, as benchmark_compressed.py --tag-legend does")
    parser.add_argument("--long-tokens",    type=int, default=DEFAULT_LONG_TOKENS, help=f"The larger answer budget in tokens (default: {DEFAULT_LONG_TOKENS})")
    return parser.parse_args()

def run_name(compressed_path: str, tag_legend: bool = False) -> str:
    """'data/compress.Frontier.p02.d001.toml' -> 'Frontier.p02.d001'; '+Legend' is added for a run with --tag-legend"""
    stem = Path(compressed_path).stem
    return (stem[len("compress."):] if stem.startswith("compress.") else stem) + ("+Legend" if tag_legend else "")

def plan_queries(data_toml: dict, compressed_tomls: dict, seed: int, tag_legend: bool = False) -> list:
    """returns every query of the benchmark runs, in the order the benchmark builds them

    compressed_tomls maps a run name to the loaded compressed file. For each run this repeats what main() of
    benchmark_compressed.py does: baselines first, then the matching versions, and for every text a freshly shuffled
    survey from one generator that is seeded anew for the run. Keep it in step with that main().
    """
    surveys, originals = bc.load_surveys(data_toml)
    queries = list()
    for name, compressed_toml in compressed_tomls.items():
        variants = bc.make_baselines(originals) + bc.select_variants(compressed_toml, surveys)
        rng = random.Random(seed)
        for vidx, v in enumerate(variants):
            for qidx, q in enumerate(bc.shuffle_survey(surveys[v["data_id"]], rng)):
                queries.append({
                    "run":          name,
                    "vidx":         vidx, # position of the text in the run, the same as in the 'results' of the benchmark file
                    "data_id":      v.get("data_id", "-"),
                    "type":         v.get("type", "Compressed Dialogue"),
                    "source_llm":   v.get("source_llm", "-"),
                    "prompt_id":    v.get("prompt_id", "-"),
                    "target_words": v.get("target_words", 0),
                    "text_tokens":  v.get("text_tokens", 0),
                    "tag_legend":   tag_legend and v.get("type") not in ("No Text", "Original Text"),
                    "qidx":         qidx,
                    "question":     q,      # {question, answers, correct_letter}
                    "text":         v["text"],
                })
    return queries

async def ask(semaphore, llm_client, text: str, q: dict, max_tokens: int, tag_legend: bool = False) -> dict:
    """asks one question about one text with the benchmark settings, returns {reply, finish, prompt_tokens}"""
    extra_body = {
        "enable_reasoning": bc.LLM_CONFIG.get("ENABLE_REASONING", True),
        "reasoning_effort": bc.LLM_CONFIG.get("REASONING_EFFORT", "medium"),
    }
    error = None
    async with semaphore:
        for _ in range(QUERY_ATTEMPTS):
            try:
                response = await llm_client.chat.completions.create(
                    model       = bc.LLM_CONFIG.get("MODEL_NAME", "-"),
                    messages    = bc.build_messages(text, q, tag_legend),
                    max_tokens  = max_tokens,
                    temperature = bc.LLM_CONFIG.get("TEMPERATURE", 0.0),
                    top_p       = bc.LLM_CONFIG.get("TOP_P", 1.0),
                    extra_body  = extra_body,
                )
                usage = getattr(response, "usage", None)
                return {"reply": response.choices[0].message.content or "", "finish": response.choices[0].finish_reason,
                        "prompt_tokens": getattr(usage, "prompt_tokens", None)}
            except Exception as e:
                error = e
    print(f"Error: A query failed {QUERY_ATTEMPTS} times: {error}", file=sys.stderr)
    return {"reply": "", "finish": "error", "prompt_tokens": None}

def row_of(query: dict, answer: dict) -> dict:
    """one line of the output: what was asked (without the long text) and what came back"""
    q = query["question"]
    row = {k: v for k, v in query.items() if k not in ("question", "text")}
    row.update({"question": q["question"], "answers": q["answers"], "expected": q["correct_letter"], **answer,
                "parsed": bc.parse_letter(answer["reply"], q["answers"]) if answer["finish"] != "error" else None})
    return row

@bc.uu.time_it
async def main() -> None:
    args = parse_arguments()
    data_path    = Path(args.data_path)
    output_path  = Path(args.output)
    bc.validate_paths([data_path, *[Path(p) for p in args.compressed_paths]], output_path)
    if args.base_url:
        bc.LLM_CONFIG["BASE_URL"] = args.base_url.strip().rstrip("/")

    queries = plan_queries(bc.load_toml(data_path), {run_name(p, args.tag_legend): bc.load_toml(Path(p)) for p in args.compressed_paths}, args.seed, args.tag_legend)
    if not queries:
        print(f"Error: Nothing to ask: no survey in '{data_path}' or no matching compressed version.", file=sys.stderr)
        sys.exit(1)
    print(f"Asking {len(queries)} queries ({len(args.compressed_paths)} run(s)), then up to {args.long_sample} letter-less replies again with {args.long_tokens} tokens.")

    bc.check_llm_server()
    semaphore  = asyncio.Semaphore(bc.LLM_CONFIG.get("CONCURRENCY", 1))
    llm_client = AsyncOpenAI(base_url=f"{bc.LLM_CONFIG['BASE_URL']}/v1", api_key=bc.LLM_CONFIG.get("API_KEY", "EMPTY"))
    max_tokens = bc.LLM_CONFIG.get("MAX_ANSWER_TOKENS", 8)
    answers = await asyncio.gather(*[ask(semaphore, llm_client, q["text"], q["question"], max_tokens, q["tag_legend"]) for q in queries])
    rows = [row_of(q, a) for q, a in zip(queries, answers)]

    # Second pass. Decision: the sample is drawn with its own generator from the seed, so it is repeatable too.
    letterless = [i for i, r in enumerate(rows) if r["parsed"] is None and r["finish"] != "error"]
    sample = sorted(random.Random(args.seed).sample(letterless, min(args.long_sample, len(letterless)))) if args.long_sample > 0 else []
    long_answers = await asyncio.gather(*[ask(semaphore, llm_client, queries[i]["text"], queries[i]["question"], args.long_tokens, queries[i]["tag_legend"]) for i in sample])
    long_rows = [{**rows[i], "long_reply": a["reply"], "long_finish": a["finish"]} for i, a in zip(sample, long_answers)]

    output = {
        "type":              "Benchmark Replies",
        "date":              str(date.today()),
        "data_path":         str(data_path),
        "compressed_paths":  [str(p) for p in args.compressed_paths],
        "seed":              args.seed,
        "tag_legend":        args.tag_legend,
        "reader_llm":        bc.LLM_CONFIG_NAME,
        "config":            dict(bc.LLM_CONFIG),
        "long_tokens":       args.long_tokens,
        "rows":              rows,
        "long":              long_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False) # one line: thousands of rows, kept small
    n_errors = sum(r["finish"] == "error" for r in rows)
    print(f"Replies: {len(rows)}, with a letter: {sum(r['parsed'] is not None for r in rows)}, without a letter: {len(letterless)}, failed: {n_errors}; asked again: {len(long_rows)}")
    print(f"Saved: '{output_path}'")

if __name__ == "__main__":
    bc.uu.print_versions(["openai", "requests"]) # for debug
    asyncio.run( main() )
