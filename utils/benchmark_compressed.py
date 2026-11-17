#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# About: A utility script that benchmarks compressed dialogues. It gives the local Gemma LLM only the compressed text
#        and checks how well it answers the single choice survey that was written for the original text.
# How to use it:
#   uv run utils/benchmark_compressed.py data/data.d001.toml data/compress.Gemma26B.toml
#   uv run utils/benchmark_compressed.py data/data.d001.toml data/compress.Gemma26B.toml --output data/benchmark.Gemma26B.d001.toml
#   uv run utils/benchmark_compressed.py data/data.d001.toml data/compress.Gemma26B.toml --base-url http://192.168.0.53:8000   # LLM on another machine
#
# The idea in short:
#   - A data file holds the original text and a survey: questions with one correct answer and a few wrong ones.
#   - A compressed file holds many shortened versions of that text (different prompts, LLMs and target lengths).
#   - A good compression keeps the facts. So we hide the original text, show the LLM one compressed version
#     and ask every survey question. The share of correct answers is the score of that compressed version.
#   - Every compressed version that belongs to the given data file is benchmarked, and the scores are printed side by side.
#   - Two reference points are printed first: the score with no text at all (the floor) and the score with the full
#     original text (the ceiling). The compressed versions follow, one line each.
import sys
import argparse
from pathlib import Path
import tomllib # for reading TOML
import tomli_w # for writing TOML
import asyncio
import random
import re
import requests
from datetime import date
import useful_utils as uu
from openai import AsyncOpenAI

DEBUG = False # poor's people debugging

# The LLM that answers the survey (the "reader").
# Decisions:
#   - Thinking is off and the answer is capped at a few tokens, because we ask for a single letter.
#     This also makes the benchmark fast: thousands of short queries.
#   - Temperature is 0, so the reader always picks its most likely answer. The benchmark should measure
#     the compressed text, not the luck of sampling, and two runs should give the same scores.
#   - No frequency/repetition penalties: they exist to improve long generated text and could only distort a one letter answer.
LLM_CONFIG_NAME = "Gemma26B_ThinkOff"
LLM_CONFIG = {
    "BASE_URL":           "http://localhost:8000",
    "MODEL_NAME":         "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
    "API_KEY":            "EMPTY",
    # LLM's parameters
    "CONCURRENCY":        32,    # serve with: --max-num-seqs 32
    "MAX_MODEL_LEN":      16384, # serve with: --max-model-len 16384
    "MAX_ANSWER_TOKENS":  8,     # a single letter is expected, leave a little room for e.g. "B)" or "**B**"
    "TEMPERATURE":        0.0,
    "TOP_P":              1.0,
    # vLLM specific (EXTRA_BODY):
    "ENABLE_REASONING":   False, # default is: True
    "REASONING_EFFORT":   "none", # default is: medium
}

# Default seed for shuffling the answers; change it with --seed to check how stable the scores are.
DEFAULT_SEED = 42

# Letters used to label the answers. The survey has 5 answers per question (1 correct + 4 wrong), the rest is a reserve.
LETTERS = "ABCDEFGHIJ"

# The conversation with the reader is built from the four messages below.
# Decision: the compressed text is not pasted next to the question. It is given earlier, as chat history, and the
# question comes as a new message. This is how compressed dialogues are meant to be used in real life: as a short
# replacement of a long chat history. So the benchmark tests them in the role they are made for.
SYSTEM_PROMPT = (
    "You answer single choice questions about a text. "
    "The user first gives you the text, and then asks a question with lettered answers. "
    "Use only the information from the given text. "
    "Reply with the letter of the correct answer and nothing else."
)
HISTORY_USER_TEMPLATE = "This is the text. Read it, I will ask you a question about it.\n\n{text}"
HISTORY_ASSISTANT     = "I have read the text. I am ready for the question."
QUESTION_TEMPLATE     = "Choose the correct answer. Reply with the letter only.\n\nQuestion: {question}\n\nAnswers:\n{answers}"

def parse_arguments() -> argparse.Namespace:
    """parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Benchmarks compressed dialogues: the Gemma LLM gets only the compressed text and answers the single choice survey from the data file. Reports the share of correct answers for every compressed version.")
    parser.add_argument("data_path",       type=str, help="Path to the data TOML file with the survey (e.g., 'data/data.d001.toml')")
    parser.add_argument("compressed_path", type=str, help="Path to the TOML file with compressed dialogues in [[results]] (e.g., 'data/compress.Gemma26B.toml')")
    parser.add_argument("--output",        type=str, default=None, help="Optional path to the output TOML file with detailed scores (must not exist)")
    parser.add_argument("--seed",          type=int, default=DEFAULT_SEED, help=f"Seed for the random order of answers (default: {DEFAULT_SEED})")
    parser.add_argument("--base-url",      type=str, default=None, help=f"Address of the LLM server, e.g. 'http://192.168.0.53:8000' (default: the BASE_URL from LLM_CONFIG, now '{LLM_CONFIG['BASE_URL']}')")
    return parser.parse_args()

def validate_paths(input_paths: list, output_path: Path = None) -> None:
    """validate that the inputs exist and the output does not"""
    for input_path in input_paths:
        if not input_path.is_file():
            print(f"Error: Input file does not exist or is not a file: '{input_path}'", file=sys.stderr)
            sys.exit(1)
    if output_path and output_path.exists():
        print(f"Error: Output file already exists: '{output_path}'", file=sys.stderr)
        sys.exit(1)

def load_toml(path: Path, what: str = None):
    if DEBUG:
        print(f"Load {what}: '{path}'")
    with open(path, "rb") as f:
        return tomllib.load(f)

def load_surveys(data_toml: dict) -> tuple[dict, dict]:
    """returns ({data_id: survey}, {data_id: original [[data]] entry}) for every [[data]] entry that has a survey

    A survey row in the data file looks like this:
        [question, correct answer, "wrong answer | wrong answer | ...", evidence]
    The evidence quotes the original text and is for humans only. It is never shown to the LLM, it would give the answer away.
    """
    surveys, originals = dict(), dict()
    for d in data_toml.get("data", []):
        data_id = d.get("data_id", None)
        questions = list()
        for row in d.get("single_choice_survey", []):
            if len(row) < 3:
                print(f"Warning: Skipping malformed survey row in '{data_id}': {row}", file=sys.stderr)
                continue
            wrong_answers = [w.strip() for w in row[2].split("|") if w.strip()]
            questions.append({"question": row[0].strip(), "correct": row[1].strip(), "wrong": wrong_answers})
        if data_id and questions:
            surveys[data_id] = questions
            originals[data_id] = d
    return surveys, originals

def shuffle_survey(survey: list, seed: int) -> list:
    """returns the survey with answers in a random order: [{question, answers, correct_letter}, ...]

    Why shuffle: in the data file the correct answer is always first. Without shuffling the LLM would score 100%
    by always answering "A", so the correct answer must land on a random letter.

    Decision: the order is drawn once per question, here, and then reused for every compressed version.
    All versions get exactly the same test (same letters), so a difference in score comes from the compressed
    text and not from a luckier order of answers. The seed makes the whole run repeatable.
    """
    rng = random.Random(seed)
    shuffled = list()
    for q in survey:
        answers = [q["correct"]] + q["wrong"]
        if len(answers) > len(LETTERS):
            print(f"Warning: Too many answers, skipping question: {q['question']}", file=sys.stderr)
            continue
        rng.shuffle(answers)
        shuffled.append({
            "question":       q["question"],
            "answers":        answers,
            "correct_letter": LETTERS[answers.index(q["correct"])],
        })
    return shuffled

def select_variants(compressed_toml: dict, surveys: dict) -> list:
    """returns the compressed versions (entries of [[results]]) that can be benchmarked with the given surveys

    A compressed file usually mixes many versions of many dialogues. Only the versions whose 'data_id' has
    a survey in the data file are taken, because questions about one dialogue say nothing about another.
    Decision: the 'type' field is not checked, some existing files do not have it in every entry.
    """
    variants, skipped = list(), 0
    for r in compressed_toml.get("results", []):
        if r.get("data_id", None) in surveys and r.get("text", None):
            variants.append(r)
        else:
            skipped += 1
    if skipped:
        print(f"Note: Skipped {skipped} compressed version(s) that do not belong to the data file (other 'data_id') or have no text.")
    return variants

def make_baselines(originals: dict) -> list:
    """returns two reference points per data file entry, in the same shape as a compressed version

    A score of a compressed text means little on its own: is 60% good or bad? Two extra runs give the scale:
      - "No Text":       the LLM gets no text at all. This is the floor: what can be guessed from the questions
                         alone (general knowledge, giveaway wording of answers). Pure luck gives 20% for 5 answers.
      - "Original Text": the LLM gets the full, uncompressed text. This is the ceiling: what this LLM can answer
                         when nothing was lost. A compressed text is good when it is close to this number, not to 100%.
    """
    baselines = list()
    for data_id, d in originals.items():
        common = {"data_id": data_id, "prompt_id": "-", "target_words": 0}
        baselines.append({**common, "type": "No Text", "source_llm": "(no text)", "text": ""})
        if d.get("text", None):
            baselines.append({**common, "type": "Original Text", "source_llm": "(original text)", "text": d["text"],
                              "text_words": d.get("text_words", 0), "text_tokens": d.get("text_tokens", 0)})
    return baselines

def check_llm_server() -> None:
    """stops the script with one clear message when the LLM server cannot be reached

    Without this check a stopped server would produce thousands of failed queries, each counted as a wrong answer,
    and the final table would look like a real (very bad) benchmark result.
    """
    base_url = LLM_CONFIG.get("BASE_URL", None)
    try:
        requests.get(f"{base_url}/v1/models", timeout=10).raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not reach the LLM service at '{base_url}'. Is the server running? ({e})", file=sys.stderr)
        sys.exit(1)

def build_messages(compressed_text: str, q: dict) -> list:
    """builds the chat: system prompt, the compressed text as history, then one question

    With an empty text (the "No Text" baseline) the history is left out, so the LLM sees only the question.
    """
    answers = "\n".join(f"{LETTERS[i]}) {a}" for i, a in enumerate(q["answers"]))
    history = [
        {"role": "user",      "content": HISTORY_USER_TEMPLATE.replace("{text}", compressed_text.strip())},
        {"role": "assistant", "content": HISTORY_ASSISTANT},
    ] if compressed_text.strip() else []
    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        *history,
        {"role": "user",      "content": QUESTION_TEMPLATE.replace("{question}", q["question"]).replace("{answers}", answers)},
    ]

def parse_letter(reply: str, n_answers: int) -> str | None:
    """extracts the chosen letter from the LLM reply, returns None if there is no clear choice

    The LLM is asked for a bare letter, but it may still decorate it, e.g. "B)", "**B**" or "Answer: B".
    The first stand-alone letter from the valid range is taken. Stand-alone means it is not a part of a word,
    so "Answer: B" gives B and not A.
    """
    if not reply:
        return None
    match = re.search(rf"(?<![A-Za-z])([{LETTERS[:n_answers]}])(?![A-Za-z])", reply.strip().upper())
    return match.group(1) if match else None

async def ask_question(semaphore, llm_client, compressed_text: str, q: dict) -> str | None:
    """asks the LLM one question, returns the chosen letter or None (no valid answer or failed query)

    Decision: every question is asked in a separate, fresh chat. Asking all questions in one growing chat would be
    closer to "a conversation", but earlier questions and their answers often reveal facts needed for later ones
    (e.g. names), so the LLM would learn from the survey itself and the score of the compressed text would be inflated.
    Separate chats also run in parallel, and the common beginning (the compressed text) is served from the vLLM prefix cache.
    """
    extra_body = { # vLLM specific parameters
        "enable_reasoning": LLM_CONFIG.get("ENABLE_REASONING", True),
        "reasoning_effort": LLM_CONFIG.get("REASONING_EFFORT", "medium"),
    }
    async with semaphore:
        try:
            response = await llm_client.chat.completions.create(
                model       = LLM_CONFIG.get("MODEL_NAME", "-"),
                messages    = build_messages(compressed_text, q),
                max_tokens  = LLM_CONFIG.get("MAX_ANSWER_TOKENS", 8),
                temperature = LLM_CONFIG.get("TEMPERATURE", 0.0),
                top_p       = LLM_CONFIG.get("TOP_P", 1.0),
                extra_body  = extra_body,
            )
            reply = response.choices[0].message.content
        except Exception as e:
            # e.g. the compressed text plus the question do not fit in the context window, or the server is down
            print(f"Error: An unexpected error occurred during LLM call: {e}", file=sys.stderr)
            return None
    if DEBUG:
        print(f"Q: {q['question']} | Expected: {q['correct_letter']} | Reply: {reply!r}")
    return parse_letter(reply, len(q["answers"]))

async def benchmark_variant(semaphore, llm_client, variant: dict, survey: list) -> dict:
    """asks all survey questions about one compressed version and returns its score

    Decision: a missing or unreadable answer counts as a wrong answer. If the LLM cannot answer from the compressed
    text, the text failed that question. Such cases are also counted separately ('no_answer'), because a big number
    usually means a technical problem (server, context window) and not a bad compression.
    """
    letters = await asyncio.gather(*[ask_question(semaphore, llm_client, variant["text"], q) for q in survey])
    wrong_questions = [i + 1 for i, (letter, q) in enumerate(zip(letters, survey)) if letter != q["correct_letter"]]
    n_correct = len(survey) - len(wrong_questions)
    return {
        # what was benchmarked: copied from the compressed file, to tell the versions apart
        "type":            variant.get("type", "Compressed Dialogue"),
        "data_id":         variant.get("data_id", "-"),
        "prompt_id":       variant.get("prompt_id", "-"),
        "source_llm":      variant.get("source_llm", "-"),
        "target_words":    variant.get("target_words", 0),
        "text_words":      variant.get("text_words", 0) or uu.count_words(variant["text"]),
        "text_tokens":     variant.get("text_tokens", 0),
        # the score
        "questions":       len(survey),
        "correct":         n_correct,
        "no_answer":       sum(letter is None for letter in letters),
        "accuracy":        round(n_correct / len(survey), 4),
        "wrong_questions": wrong_questions, # numbers of the questions (from 1, in survey order) that were answered wrong
    }

def print_header() -> None:
    header = f"{'data_id':<8} {'prompt':<7} {'source_llm':<24} {'target':>6} {'words':>6} {'tokens':>6} {'correct':>9} {'accuracy':>8} {'no answer':>9}"
    print(header)
    print("-" * len(header))

def print_score(s: dict) -> None:
    """prints one line: the score of one text (a baseline or a compressed version)"""
    correct = f"{s['correct']}/{s['questions']}"
    print(f"{s['data_id']:<8} {s['prompt_id']:<7} {s['source_llm']:<24} {s['target_words']:>6} {s['text_words']:>6} {s['text_tokens']:>6} {correct:>9} {s['accuracy']:>8.1%} {s['no_answer']:>9}", flush=True)

@uu.time_it
async def main() -> None:
    args = parse_arguments()
    data_path       = Path( args.data_path )
    compressed_path = Path( args.compressed_path )
    output_path     = Path( args.output ) if args.output else None
    validate_paths([data_path, compressed_path], output_path)
    if args.base_url:
        # Decision: the address is taken from the command line, not hardcoded for one machine. It overrides LLM_CONFIG,
        # so the check of the server, the client and the saved 'configs' in the --output file all use the same address.
        LLM_CONFIG["BASE_URL"] = args.base_url.strip().rstrip("/")

    # load the surveys and draw the order of answers (once, shared by all compressed versions)
    surveys, originals = load_surveys(load_toml(data_path, "data with survey"))
    if not surveys:
        print(f"Error: No 'single_choice_survey' found in: '{data_path}'", file=sys.stderr)
        sys.exit(1)
    surveys = {data_id: shuffle_survey(survey, args.seed) for data_id, survey in surveys.items()}

    # pick the compressed versions that match the surveys
    variants = select_variants(load_toml(compressed_path, "compressed dialogues"), surveys)
    if not variants:
        print(f"Error: No compressed dialogue for data_id {sorted(surveys)} found in: '{compressed_path}'", file=sys.stderr)
        sys.exit(1)
    # the reference points go first: they are the scale for reading all the lines that follow
    baselines = make_baselines(originals)
    n_compressed = len(variants)
    variants = baselines + variants
    n_queries = sum(len(surveys[v["data_id"]]) for v in variants)
    print(f"Benchmarking {len(baselines)} baseline(s) and {n_compressed} compressed version(s), {n_queries} LLM queries.")

    # query the LLM; all versions are started together and the semaphore keeps the load at what the server can take
    check_llm_server()
    concurrency = LLM_CONFIG.get("CONCURRENCY", 1)
    print(f"Querying LLM: {LLM_CONFIG_NAME} ({LLM_CONFIG['BASE_URL']}), Concurrency: {concurrency}")
    semaphore  = asyncio.Semaphore(concurrency)
    llm_client = AsyncOpenAI(base_url=f"{LLM_CONFIG['BASE_URL']}/v1", api_key=LLM_CONFIG.get("API_KEY", "EMPTY"))
    # Decision: everything is started at once (the semaphore limits the real load), but the lines are printed
    # in a fixed order: baselines, then compressed versions as in the compressed file. Each line appears as soon as
    # it and all lines before it are ready, so a long run shows progress and two runs are easy to compare.
    tasks = [asyncio.create_task(benchmark_variant(semaphore, llm_client, v, surveys[v["data_id"]])) for v in variants]
    print_header()
    scores = list()
    for task in tasks:
        scores.append(await task)
        print_score(scores[-1])
    n_no_answer = sum(s["no_answer"] for s in scores)
    if n_no_answer:
        print(f"Warning: {n_no_answer} of {n_queries} queries gave no valid answer (counted as wrong). Check the errors above.", file=sys.stderr)

    # save output TOML file (optional), with everything needed to repeat or analyze the run
    if output_path:
        output_toml = {
            'configs': {LLM_CONFIG_NAME: LLM_CONFIG},
            'benchmark': {
                'type':            "Benchmark Compressed Dialogue",
                'date':            date.today(),
                'data_path':       str(data_path),
                'compressed_path': str(compressed_path),
                'reader_llm':      LLM_CONFIG_NAME,
                'seed':            args.seed,
            },
            'results': scores,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            tomli_w.dump(output_toml, f, multiline_strings=True)
        print(f"Saved: '{output_path}'")

if __name__ == "__main__":
    uu.print_versions(["openai", "requests"]) # for debug
    asyncio.run( main() )
