#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# About: A utility script that use local Gemma LLM to compresses text from [[data]] and outputs the results to a structured TOML file.
# How to use it:
#   uv run utils/compress_Gemma26B.py data/prompts.compress.toml "data/data.*.toml" data/compress.Gemma26B.toml
import sys
import argparse
from pathlib import Path
import tomllib # for reading TOML
import tomli_w # for writing TOML
import asyncio
import glob
from datetime import date
import useful_utils as uu
from openai import OpenAI, AsyncOpenAI

DEBUG = False # poor's people debugging

# word count target budget (after compression), the actual word count in the results may be slightly different
TARGET_WORDS = [1200, 1100, 1000, 900, 800, 700, 600, 500, 400, 350, 300, 250, 200, 150, 100]

# use the model of your choice
LLM_CONFIGS = {
    "Gemma26B_ThinkOff": {
        "BASE_URL":           "http://localhost:8000",
        "MODEL_NAME":         "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
        "API_KEY":            "EMPTY",
        # LLM's parameters
        "CONCURRENCY":        32,    # serve with: --max-num-seqs 32
        "MAX_MODEL_LEN":      16384, # serve with: --max-model-len 16384
        "TEMPERATURE":        0.3,
        "TOP_P":              0.95,
        "FREQUENCY_PENALTY":  0.2,
        "PRESENCE_PENALTY":   0.0,
        # vLLM specific (EXTRA_BODY):
        "MIN_P":              0.05,
        "REPETITION_PENALTY": 1.07,
        "ENABLE_REASONING":   False, # default is: True
        "REASONING_EFFORT":   "none", # default is: medium
    },
    "Gemma26B_ThinkMedium": {
        "BASE_URL":           "http://localhost:8000",
        "MODEL_NAME":         "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
        "API_KEY":            "EMPTY",
        # LLM's parameters
        "CONCURRENCY":        32,    # serve with: --max-num-seqs 32
        "MAX_MODEL_LEN":      16384, # serve with: --max-model-len 16384
        "TEMPERATURE":        0.3,
        "TOP_P":              0.95,
        "FREQUENCY_PENALTY":  0.2,
        "PRESENCE_PENALTY":   0.0,
        # vLLM specific (EXTRA_BODY):
        "MIN_P":              0.05,
        "REPETITION_PENALTY": 1.07,
        "ENABLE_REASONING":   True, # pick default value
        "REASONING_EFFORT":   "medium", # pick default value
    },
}

# tags
TAG_TARGET_WORDS = '{target_words}'
TAG_DIALOGUE     = '{dialogue}'

def parse_arguments() -> argparse.Namespace:
    """parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Compresses text from [[data]] (from multiple TOML files matching a glob pattern) using the Gemma LLM and outputs the results to a structured TOML file.")
    parser.add_argument("prompts_path", type=str, help="Path to the prompts in TOML file (e.g., 'data/prompts.compress.toml')")
    parser.add_argument("glob_pattern", type=str, help="Glob pattern for input TOML files (e.g., 'data/data.*.toml')")
    parser.add_argument("output_path",  type=str, help="Path to the output TOML file (e.g., 'compress.results.toml')")
    return parser.parse_args()

def validate_paths(input_path: Path, output_path: Path) -> None:
    """validate that the input exists and the output does not"""
    if not input_path.exists():
        print(f"Error: Input file does not exist: '{input_path}'", file=sys.stderr)
        sys.exit(1)
    if not input_path.is_file():
        print(f"Error: Input path is not a valid file: '{input_path}'", file=sys.stderr)
        sys.exit(1)
    if output_path.exists():
        print(f"Error: Output file already exists: '{output_path}'", file=sys.stderr)
        sys.exit(1)

def load_toml(path: Path, what: str = None):
    if DEBUG:
        print(f"Load {what}: '{path}'")
    with open(path, "rb") as f:
        return tomllib.load(f)

def prepare_prompt(prompt_template: str, dialogue: str, n_words: int) -> str:
    return prompt_template.replace(TAG_DIALOGUE, dialogue).replace(TAG_TARGET_WORDS, str(n_words))

def iterate_inputs(paths: list):
    for path in paths:
        tml = load_toml(path, "input data")
        for d in tml.get("data", []):
            if (d.get("type", None) == "Dialogue"):
                yield d.get("data_id", None), d.get("text", None)

def iterate_prompts(prompts_toml: dict, prompt_type: str):
    for prompt in prompts_toml.get('prompts', []):
        if (
            prompt.get("type", "") == prompt_type
            and "prompt_id" in prompt
            and "system_prompt_template" in prompt
            and "user_prompt_template" in prompt
        ):
            yield prompt

def iterate_llms(llm_configs: dict):
    yield from llm_configs.items()

async def process_with_llm(semaphore, llm_client, llm_config: dict, prompt: dict, data_text: str, n_words: int):
    result_str    = None
    # prepare prompts
    system_prompt = prepare_prompt(prompt["system_prompt_template"], data_text, n_words)
    user_prompt   = prepare_prompt(prompt["user_prompt_template"],   data_text, n_words)
    # get number of output tokens, query LLM to count input tokens
    input_tokens  = uu.get_precise_token_count(
        llm_config.get("BASE_URL", None),
        llm_config.get("MODEL_NAME", None),
        system_prompt,
        user_prompt,
    )
    output_tokens = llm_config.get("MAX_MODEL_LEN", 0) - input_tokens - 1 # one token safety margin
    # vLLM specific parameters
    extra_body = {
        "min_p":              llm_config.get("MIN_P", 0.0),
        "repetition_penalty": llm_config.get("REPETITION_PENALTY", 0.0),
        "enable_reasoning":   llm_config.get("ENABLE_REASONING", True),
        "reasoning_effort":   llm_config.get("REASONING_EFFORT", "medium"),
    }
    # query LLM to compress text
    async with semaphore:
        try:
            response = await llm_client.chat.completions.create(
                model=llm_config.get("MODEL_NAME", "-"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                max_tokens        = output_tokens,
                temperature       = llm_config.get("TEMPERATURE", 1.0),
                top_p             = llm_config.get("TOP_P", 1.0),
                frequency_penalty = llm_config.get("FREQUENCY_PENALTY", 0.0),
                presence_penalty  = llm_config.get("PRESENCE_PENALTY", 0.0),
                extra_body        = extra_body, # vLLM specific, min_p, reasoning, ..
            )
            # TBD: not sure how to skip reasoning/thinking output
            result_str = response.choices[0].message.content.strip() # take results
        except Exception as e:
            print(f"Error: An unexpected error occurred during LLM call: {e}", file=sys.stderr)
            result_str = None
    return result_str

async def process_one(semaphore, llm_client, prompt, data_id: str, data_text: str, n_words: int, llm_config_name: str, llm_config, llm_results):
    # start compressing with the LLM
    if DEBUG:
        print(f"Query LLM: {llm_config_name}, Prompt: {prompt['prompt_id']}, Data: {data_id}, T.Words: {n_words}")
    compress_text = await process_with_llm(semaphore, llm_client, llm_config, prompt, data_text, n_words)
    # capture results
    if compress_text:
        d = dict()
        d['type']         = "Compressed Dialogue"
        d['date']         = date.today()
        d['prompt_id']    = prompt["prompt_id"]
        d['data_id']      = data_id
        d['source_llm']   = llm_config_name
        d['target_words'] = n_words
        d['text']         = compress_text
        llm_results.append(d)

async def process_all(semaphore, llm_client, prompts_toml, input_files, llm_config_name, llm_config, llm_results):
    tasks = set() # use a set to keep track of currently running tasks
    # iterate all prompts
    for prompt in iterate_prompts(prompts_toml, "Compress Dialogue"):
        # iterate each input file
        for data_id, data_text in iterate_inputs(input_files):
            # iterate over all target word counts
            for n_words in TARGET_WORDS:
                # create the task
                task = asyncio.create_task(
                    process_one(semaphore, llm_client, prompt, data_id, data_text, n_words, llm_config_name, llm_config, llm_results)
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
    prompts_path = Path( args.prompts_path )
    output_path  = Path( args.output_path )
    validate_paths(prompts_path, output_path)

    # load prompts
    prompts_toml = load_toml(prompts_path, "prompts")
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

        llm_client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key=llm_config.get("API_KEY", "EMPTY"))
        await process_all(semaphore, llm_client, prompts_toml, input_files, llm_config_name, llm_config, llm_results)

    # save output TOML file
    output_toml = {
        'configs': LLM_CONFIGS,
        'results': llm_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        tomli_w.dump(output_toml, f, multiline_strings=True)

if __name__ == "__main__":
    uu.print_versions(["openai", "requests"]) # for debug
    asyncio.run( main() )
