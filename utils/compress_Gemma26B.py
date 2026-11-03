#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# About: A utility script that compresses text from [[data]] using the Gemma LLM and outputs the results to a structured TOML file.
# How to use it:
#   uv run utils/compress_Gemma26B.py data/prompts.compress.toml "data/data.*.toml" data/compress.Gemma26B.toml
import sys
import argparse
from pathlib import Path
import tomllib # for reading TOML
import tomli_w # for writing TOML
import requests
from requests.exceptions import ConnectionError, HTTPError
import asyncio
from openai import OpenAI, AsyncOpenAI
import glob
from datetime import date

# word count target budget (after compression), final word count may be different
TARGET_WORDS = [1200, 1100, 1000, 900, 800, 700, 600, 500, 400, 350, 300, 250, 200, 150, 100]

# use the model of your choice
LLM_CONFIGS = {
    "Gemma26B_ThinkOff": {
        "BASE_URL":           "http://localhost:8000",
        "MODEL_NAME":         "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
        "API_KEY":            "EMPTY",
        # LLM's parameters
        "CONCURRENCY":        32,   # serve with: --max-num-seqs 32
        "MAX_MODEL_LEN":      8192, # serve with: --max-model-len 8192
        "TEMPERATURE":        0.3,
        "TOP_P":              0.95,
        "FREQUENCY_PENALTY":  0.2,
        "PRESENCE_PENALTY":   0.0,
        # vLLM specific (EXTRA_BODY):
        "MIN_P":              0.05,
        "REPETITION_PENALTY": 1.07,
        "ENABLE_REASONING":   False, # default: True
        "REASONING_EFFORT":   "none", # default: medium
    },
    "Gemma26B_ThinkMedium": {
        "BASE_URL":           "http://localhost:8000",
        "MODEL_NAME":         "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
        "API_KEY":            "EMPTY",
        # LLM's parameters
        "CONCURRENCY":        32,   # serve with: --max-num-seqs 32
        "MAX_MODEL_LEN":      8192, # serve with: --max-model-len 8192
        "TEMPERATURE":        0.3,
        "TOP_P":              0.95,
        "FREQUENCY_PENALTY":  0.2,
        "PRESENCE_PENALTY":   0.0,
        # vLLM specific (EXTRA_BODY):
        "MIN_P":              0.05,
        "REPETITION_PENALTY": 1.07,
        "ENABLE_REASONING":   True, # default value
        "REASONING_EFFORT":   "medium", # default value
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
    if what:
        print(f"Load {what}: '{path}'") # show progress
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

def process_with_llm(llm_config: dict, prompt: dict, data_text: str, n_words: int):
    system_prompt = prepare_prompt(prompt["system_prompt_template"], data_text, n_words)
    user_prompt   = prepare_prompt(prompt["user_prompt_template"],   data_text, n_words)
    # TBD: use LLM!
    # "EXTRA_BODY": {
    #     # vLLM specific parameters
    #     "min_p":              MIN_P,
    #     "repetition_penalty": REPETITION_PENALTY,
    #     "enable_reasoning":   ENABLE_REASONING,
    #     "reasoning_effort":   REASONING_EFFORT,
    # },
    return 'xxx'

def main() -> None:
    args = parse_arguments()
    prompts_path = Path( args.prompts_path )
    output_path  = Path( args.output_path )
    validate_paths(prompts_path, output_path)

    # find all files matching the glob pattern
    input_files = sorted(glob.glob(args.glob_pattern))
    if not input_files:
        print(f"Warning: No files matched the glob pattern: {args.glob_pattern}")
        return

    # start with empty results
    output_toml = {}
    output_toml['results'] = list()
    try:
        # load prompts
        prompts_toml = load_toml(prompts_path, "prompts")
        # iterate all LLM configs
        for llm_config_name, llm_config in iterate_llms(LLM_CONFIGS):
            # iterate all prompts
            for prompt in iterate_prompts(prompts_toml, "Compress Dialogue"):
                # iterate each input file
                for data_id, data_text in iterate_inputs(input_files):
                    # iterate over all target word counts
                    for n_words in TARGET_WORDS:
                        # start compressing with the LLM
                        print(f"Use LLM: {llm_config_name}, Prompt: {prompt['prompt_id']}, Data: {data_id}, T.Words: {n_words}") # show progress
                        compress_text = process_with_llm(llm_config, prompt, data_text, n_words)
                        # capture results
                        d = dict()
                        d['type']         = "Compressed Dialogue"
                        d['date']         = date.today()
                        d['prompt_id']    = prompt["prompt_id"]
                        d['data_id']      = data_id
                        d['source_llm']   = llm_config_name
                        d['target_words'] = n_words
                        d['text']         = compress_text
                        output_toml['results'].append(d)
        # save output TOML file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            tomli_w.dump(output_toml, f, multiline_strings=True)

    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
