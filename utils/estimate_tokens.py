#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# About: A utility script that enhances a TOML file by automatically calculating and appending word counts and LLM token counts for text entries in [[data]] and [[results]].
# How to use it:
#   uv run python utils/estimate_tokens.py <input_path> <output_path>
import sys
import argparse
from pathlib import Path
import tomllib # for reading TOML
import tomli_w # for writing TOML
import useful_utils as uu

# use the model of your choice
BASE_URL   = "http://localhost:8000"
MODEL_NAME = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"

def parse_arguments() -> argparse.Namespace:
    """parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Enhances a TOML file by calculating and appending word and LLM token counts for text entries in [[data]] & [[results]].")
    parser.add_argument("input_path",  type=Path, help="Path to the input TOML file.")
    parser.add_argument("output_path", type=Path, help="Path to the output TOML file.")
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
    input_path  = Path( args.input_path )
    output_path = Path( args.output_path )
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
                    item["text_words"]  = uu.count_words(item["text"])
                    item["text_tokens"] = uu.get_precise_token_count(
                        BASE_URL,
                        MODEL_NAME,
                        '',           # system prompt
                        item["text"], # user prompt
                    )

        # save output TOML file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            tomli_w.dump(toml_data, f, multiline_strings=True)

    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
