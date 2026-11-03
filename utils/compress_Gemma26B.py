#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# About: A utility script that compresses text from [[data]] using the Gemma LLM and outputs the results to a structured TOML file.
# How to use it:
#   uv run utils/compress_Gemma26B.py data/prompts.compress.toml "data/data.*.toml" compress.Gemma26B.toml
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

TARGET_WORDS = [1200, 1100, 1000, 900, 800, 700, 600, 500, 400, 350, 300, 250, 200, 150, 100]

# use the model of your choice
BASE_URL   = "http://localhost:8000"
MODEL_NAME = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
API_KEY    = "EMPTY"
# LLM's parameters
CONCURRENCY    = 32   # serve with: --max-num-seqs 32
MAX_MODEL_LEN  = 8192 # serve with: --max-model-len 8192
TEMPERATURE    = 0.3
TOP_P          = 0.95
FREQ_PENALTY   = 0.2 # frequency penalty
PRES_PENALTY   = 0.0 # presence penalty
# vLLM specific (EXTRA_BODY):
MIN_P            = 0.05
REPE_PENALTY     = 1.07 # repetition penalty
ENABLE_REASONING = False # default: True
REASONING_EFFORT = "none" # default: medium
EXTRA_BODY = {
    # vLLM specific parameters
    "min_p":              MIN_P,
    "repetition_penalty": REPE_PENALTY,
    "enable_reasoning":   ENABLE_REASONING,
    "reasoning_effort":   REASONING_EFFORT,
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

def main() -> None:
    args = parse_arguments()
    prompts_path = Path( args.prompts_path )
    output_path  = Path( args.output_path )
    validate_paths(prompts_path, output_path)

    # find all files matching the glob pattern
    input_files = glob.glob(args.glob_pattern)
    if not input_files:
        print(f"Warning: No files matched the glob pattern: {args.glob_pattern}")
        return

    try:
        # load prompts
        with open(prompts_path, "rb") as f:
            prompts_toml = tomllib.load(f)
        # iterate all prompts
        output_toml = {}
        output_toml['results'] = list()
        for prompt in prompts_toml.get('prompts', []):
            # take the prompt
            if (
                prompt.get("type", "") == "Compress Dialogue"
                and "prompt_id" in prompt
                and "system_prompt_template" in prompt
                and "user_prompt_template" in prompt
            ):
                for n in TARGET_WORDS:
                    data_id   = 'd00'
                    data_text = 'xxx'
                    system_prompt = prompt["system_prompt_template"].replace(TAG_TARGET_WORDS, str(n)).replace(TAG_DIALOGUE, data_text)
                    user_prompt   = prompt["user_prompt_template"].replace(TAG_TARGET_WORDS, str(n)).replace(TAG_DIALOGUE, data_text)

                    # TBD: compress
                    compress_text = 'xxx'

                    # prepare result item:
                    d = dict()
                    d['type']         = "Compressed Dialogue"
                    d['date']         = date.today()
                    d['prompt_id']    = prompt["prompt_id"]
                    d['data_id']      = data_id
                    d['source_llm']   = "Gemma26B"
                    d['target_words'] = n
                    d['text']         = compress_text
                    output_toml['results'].append(d)

        # process each input file
        for file_path in sorted(input_files):
            print("->", file_path)
            #try:
            #    with open(file_path, "rb") as f:
            #        data = tomllib.load(f)
            #
            #    # Expecting structure: [[data]]\n text = "..."
            #    if isinstance(data, dict) and "data" in data:
            #        items = data["data"]
            #        if isinstance(items, list):
            #            for item in items:
            #                if isinstance(item, dict) and "text" in item:
            #                    collected_texts.append(item["text"])
            #except Exception as e:
            #    print(f"Error reading {file_path}: {e}")

        # save output TOML file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            tomli_w.dump(output_toml, f, multiline_strings=True)

    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
