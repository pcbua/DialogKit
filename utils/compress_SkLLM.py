#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# About: A utility script that use Scikit to compresses text from [[data]] and outputs the results to a structured TOML file.
# Inspired by: https://www.kdnuggets.com/using-llm-summarization-in-your-scikit-learn-pipelines-with-scikit-llm
# How to use it:
#   TQDM_DISABLE=1 uv run --with 'scikit-llm>=1.4' utils/compress_SkLLM.py "data/data.*.toml" data/compress.SkLLM.toml
import sys
import argparse
from pathlib import Path
import tomllib # for reading TOML
import tomli_w # for writing TOML
import asyncio
import glob
from datetime import date
import useful_utils as uu
from skllm.config import SKLLMConfig
from skllm.models.gpt.text2text.summarization import GPTSummarizer

DEBUG = False # poor's people debugging

# word count target budget (after compression), the actual word count in the results may be slightly different
TARGET_WORDS = [1200, 1100, 1000, 900, 800, 700, 600, 500, 400, 350, 300, 250, 200, 150, 100]

# use the model of your choice
LLM_CONFIGS = {
    "Gemma26B": {
        "BASE_URL":    "http://localhost:8000",
        "MODEL_NAME":  "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
        "API_KEY":     "EMPTY",
        "CONCURRENCY": 32, # serve with: --max-num-seqs 32
    },
}

def parse_arguments() -> argparse.Namespace:
    """parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Compresses text from [[data]] (from multiple TOML files matching a glob pattern) using the Scikit and outputs the results to a structured TOML file.")
    parser.add_argument("glob_pattern", type=str, help="Glob pattern for input TOML files (e.g., 'data/data.*.toml')")
    parser.add_argument("output_path",  type=str, help="Path to the output TOML file (e.g., 'compress.results.toml')")
    return parser.parse_args()

def validate_paths(output_path: Path) -> None:
    """validate that the output path not exists"""
    if output_path.exists():
        print(f"Error: Output file already exists: '{output_path}'", file=sys.stderr)
        sys.exit(1)

def load_toml(path: Path, what: str = None):
    if DEBUG:
        print(f"Load {what}: '{path}'")
    with open(path, "rb") as f:
        return tomllib.load(f)

def iterate_inputs(paths: list):
    for path in paths:
        tml = load_toml(path, "input data")
        for d in tml.get("data", []):
            if (d.get("type", None) == "Dialogue"):
                yield d.get("data_id", None), d.get("text", None)

def iterate_llms(llm_configs: dict):
    yield from llm_configs.items()

async def process_with_llm(semaphore, model_name: str, data_text: str, n_words: int):
    result_str = None
    # use skllm to compress text
    async with semaphore:
        try:
            llm_summarizer = GPTSummarizer(model=f"custom_url::{model_name}", max_words=n_words)
            # run the blocking scikit-learn workflow in a separate thread pool
            response = await asyncio.to_thread( llm_summarizer.fit_transform, [data_text] ) # run on a single document
            if response and len(response) > 0:
                result_str = response[0]
        except Exception as e:
            print(f"Error: An unexpected error occurred during LLM call: {e}", file=sys.stderr)
            result_str = None
    return result_str

async def process_one(semaphore, data_id: str, data_text: str, n_words: int, llm_config_name: str, llm_config, llm_results):
    # start compressing with the LLM
    if DEBUG:
        print(f"Query LLM: {llm_config_name}, Data: {data_id}, T.Words: {n_words}")
    model_name = llm_config.get("MODEL_NAME", None)
    compress_text = await process_with_llm(semaphore, model_name, data_text, n_words)
    # capture results
    if compress_text:
        d = dict()
        d['type']         = "Compressed Dialogue"
        d['date']         = date.today()
        d['prompt_id']    = "NA"
        d['data_id']      = data_id
        d['source_llm']   = llm_config_name
        d['target_words'] = n_words
        d['text']         = compress_text
        llm_results.append(d)

async def process_all(semaphore, input_files, llm_config_name, llm_config, llm_results):
    tasks = set() # use a set to keep track of currently running tasks
    # iterate each input file
    for data_id, data_text in iterate_inputs(input_files):
        # iterate over all target word counts
        for n_words in TARGET_WORDS:
            # create the task
            task = asyncio.create_task(
                process_one(semaphore, data_id, data_text, n_words, llm_config_name, llm_config, llm_results)
            )
            tasks.add(task)
            task.add_done_callback(tasks.discard) # a callback to remove task from set when done
            # if the que is full, wait for at least one to finish
            if len(tasks) >= 2 * semaphore.limit:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    # wait for the final remaining tasks to wrap up
    if tasks:
        await asyncio.gather(*tasks)

async def main() -> None:
    args = parse_arguments()
    output_path  = Path( args.output_path )
    validate_paths(output_path)

    # find all files matching the glob pattern
    input_files = sorted(glob.glob(args.glob_pattern))
    if not input_files:
        print(f"Warning: No files matched the glob pattern: {args.glob_pattern}")
        return

    # iterate all LLM configs
    llm_results = list() # start with empty list
    for llm_config_name, llm_config in iterate_llms(LLM_CONFIGS):
        base_url    = llm_config.get("BASE_URL", None)
        model_name  = llm_config.get("MODEL_NAME", None)
        if not base_url or not model_name:
            print(f"Error: Problem with LLM; Base URL: {base_url}; Model name: {model_name}", file=sys.stderr)
            continue
        concurrency = llm_config.get("CONCURRENCY", 1)
        print(f"Querying LLM: {llm_config_name} ({base_url}), Concurrency: {concurrency}")
        semaphore = asyncio.Semaphore(concurrency)
        semaphore.limit = concurrency

        url = f"{base_url}/v1"
        SKLLMConfig.set_gpt_url(url) # depending on version: SKLLMConfig.set_gpt_url(), SKLLMConfig.set_openai_base_url(), openai.api_base = url, SKLLMConfig.set_openai_backend()
        SKLLMConfig.set_openai_key(llm_config.get("API_KEY", "EMPTY"))
        await process_all(semaphore, input_files, llm_config_name, llm_config, llm_results)

    # save output TOML file
    output_toml = {
        'configs': LLM_CONFIGS,
        'results': llm_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        tomli_w.dump(output_toml, f, multiline_strings=True)

if __name__ == "__main__":
    uu.print_versions(["skllm", "openai", "requests"]) # for debug
    asyncio.run( main() )
