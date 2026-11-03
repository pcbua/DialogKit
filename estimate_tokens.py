#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# About: A tool to enhance a TOML file by calculating and appending word and LLM token counts for text entries in [[data]] & [[results]].
# How to use it:
#   uv run python estimate_tokens.py <INPUT_TOML> <OUTPUT_TOML>
import sys
import argparse
from pathlib import Path
import requests
from requests.exceptions import ConnectionError, HTTPError
import re
import tomllib # for reading TOML
import tomli_w # for writing TOML

# use the model of your choice
BASE_URL   = "http://localhost:8000"
MODEL_NAME = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"

def count_words(text: str) -> int:
    """counts words by splitting on whitespace and common punctuation/symbols that LLM tokenizers typically break apart"""
    if not text:
        return 0
    normalized_text = re.sub(r'[/:\-\s]+', ' ', text) # replace slashes, colons, hyphens, and whitespace with a single space
    words = normalized_text.split() # split by spaces to get the individual words
    return len(words)

def get_precise_token_count(system_content: str, user_content: str) -> int:
    """calculates the exact number of tokens as the model sees them, including the system prompt and user prompt"""
    try:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content}
        ]
        response = requests.post(
            f"{BASE_URL}/tokenize",
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "add_generation_prompt": True,
            },
            timeout=30,
        )
        response.raise_for_status()
    except ConnectionError:
        print(f"Error: Could not connect to the LLM service at '{BASE_URL}'. Is the server running?", file=sys.stderr)
        sys.exit(1)
    except HTTPError as e:
        print(f"HTTP Error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    return response.json()["count"]

def parse_arguments() -> argparse.Namespace:
    """parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Enhances a TOML file by calculating and appending word and LLM token counts for text entries in [[data]] & [[results]].")
    parser.add_argument("INPUT_TOML", type=Path, help="Path to the input TOML file.")
    parser.add_argument("OUTPUT_TOML", type=Path, help="Path to the output TOML file.")
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
    input_path:  Path = args.INPUT_TOML
    output_path: Path = args.OUTPUT_TOML
    validate_paths(input_path, output_path)
    try:
        # load input TOML
        print(f"Converting: {input_path} => {output_path}")
        with open(input_path, "rb") as f:
            toml_data = tomllib.load(f)

        # iterate all text entries of [[data]] & [[results]]
        for section in ("data", "results"):
            for item in toml_data.get(section, []):
                if "text" in item:
                    item["text_words"]  = count_words(item["text"])
                    item["text_tokens"] = get_precise_token_count('', item["text"])

        # save to output path
        with open(output_path, "wb") as f:
            tomli_w.dump(toml_data, f, multiline_strings=True)

    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
