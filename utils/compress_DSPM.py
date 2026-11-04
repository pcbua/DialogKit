#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# About: A utility script that use DSPM to compresses text from [[data]] and outputs the results to a structured TOML file.
# Inspired by: https://github.com/zatchbell1311-wq/Kernl
# How to use it:
#   uv run --with 'dspm-memory>=0.2.4' utils/compress_DSPM.py "data/data.*.toml" data/compress.DSPM.toml
import sys
import argparse
from pathlib import Path
import tomllib # for reading TOML
import tomli_w # for writing TOML
import asyncio
import glob
from datetime import date
import useful_utils as uu
from dspm import DSPMMemory
from openai import OpenAI, AsyncOpenAI

DEBUG = True # poor's people debugging

# word count target budget (after compression), the actual word count in the results may be slightly different
TARGET_WORDS = [1200, 1100, 1000, 900, 800, 700, 600, 500, 400, 350, 300, 250, 200, 150, 100]

# use the model of your choice
LLM_CONFIGS = {
    "Gemma26B": {
        "BASE_URL":    "http://localhost:8000",
        "MODEL_NAME":  "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
        "API_KEY":     "EMPTY",
        "CONCURRENCY": 1, # DSPM is sequential :(
    },
}

def parse_arguments() -> argparse.Namespace:
    """parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Compresses text from [[data]] (from multiple TOML files matching a glob pattern) using the DSPM and outputs the results to a structured TOML file.")
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

async def process_with_llm(semaphore, llm_client, model_name: str, data_text: str, n_words: int):
    result_str = None
    # use DSPM to compress text
    async with semaphore:
        try:
            dspm_mem = DSPMMemory(
                budget=round(n_words * 1.625), # estimated number of tokens, based on Gemma 4 26B stats
                llm_client=llm_client,
                model=model_name,
            )
            # extract dialogue lines (turns) from data_text
            for line in data_text.split('\n'):
                if ":" in line:
                    role, text = line.split(":", 1)
                    if role.upper() == "AGENT":
                        dspm_mem.add_turn("assistant", text) # not sure if "Agent:" should be skipped here; text - skipped, line - included
                    else:
                        dspm_mem.add_turn("user", line) # we have to take the whole line because dialogues can have more than two participants
            # extract compressed prompt
            result_str = dspm_mem.get_context(query="What are the rules?") # DSPM accepts natural language
        except Exception as e:
            print(f"Error: An unexpected error occurred during DSPM call: {e}", file=sys.stderr)
            result_str = None
    return result_str

async def process_one(semaphore, llm_client, data_id: str, data_text: str, n_words: int, llm_config_name: str, llm_config, llm_results):
    # start compressing with the DSPM
    if DEBUG:
        print(f"Query DSPM: {llm_config_name}, Data: {data_id}, T.Words: {n_words}")
    model_name = llm_config.get("MODEL_NAME", None)
    compress_text = await process_with_llm(semaphore, llm_client, model_name, data_text, n_words)
    # capture results
    if compress_text:
        d = dict()
        d['type']         = "Compressed Dialogue"
        d['date']         = date.today()
        d['prompt_id']    = "NA"
        d['data_id']      = data_id
        d['source_llm']   = f"DSPM/{llm_config_name}"
        d['target_words'] = n_words
        d['text']         = compress_text
        llm_results.append(d)

async def process_all(semaphore, llm_client, input_files, llm_config_name, llm_config, llm_results):
    tasks = set() # use a set to keep track of currently running tasks
    # iterate each input file
    for data_id, data_text in iterate_inputs(input_files):
        # iterate over all target word counts
        for n_words in TARGET_WORDS:
            # create the task
            task = asyncio.create_task(
                process_one(semaphore, llm_client, data_id, data_text, n_words, llm_config_name, llm_config, llm_results)
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
            print(f"Error: Problem with LLM config; Base URL: {base_url}; Model name: {model_name}", file=sys.stderr)
            continue
        concurrency = llm_config.get("CONCURRENCY", 1)
        print(f"Querying LLM: {llm_config_name} ({base_url}), Concurrency: {concurrency}")
        semaphore = asyncio.Semaphore(concurrency)
        semaphore.limit = concurrency

        llm_client = OpenAI(base_url=f"{base_url}/v1", api_key=llm_config.get("API_KEY", "EMPTY")) # TBD: later update to AsyncOpenAI()
        await process_all(semaphore, llm_client, input_files, llm_config_name, llm_config, llm_results)

    # save output TOML file
    output_toml = {
        'configs': LLM_CONFIGS,
        'results': llm_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        tomli_w.dump(output_toml, f, multiline_strings=True)

if __name__ == "__main__":
    uu.print_versions(["dspm", "openai", "requests"]) # for debug
    asyncio.run( main() )
