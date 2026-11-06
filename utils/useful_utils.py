# -*- coding: utf-8 -*-
# About: A library of things repeated in other scripts.
# How to use it (in Python):
#   import useful_utils as uu
import sys
import time
import inspect
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar, Union
import importlib, importlib.metadata
import requests
from requests.exceptions import ConnectionError, HTTPError
import re

F = TypeVar("F", bound=Callable[..., Any])

def time_it(func: F) -> F:
    """decorator to measure and print the execution time of a function"""
    def _print_time(elapsed: float) -> None:
        if elapsed < 60:
            print(f"[{func.__name__}] Execution time: {elapsed:.3f} seconds")
        elif elapsed < 3600:
            minutes = elapsed / 60
            print(f"[{func.__name__}] Execution time: {minutes:.2f} minutes")
        else:
            hours = elapsed / 3600
            print(f"[{func.__name__}] Execution time: {hours:.2f} hours")

    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                end_time = time.perf_counter()
                _print_time(end_time - start_time)
        return async_wrapper  # type: ignore
    else:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                end_time = time.perf_counter()
                _print_time(end_time - start_time)
        return sync_wrapper  # type: ignore

def count_words(text: str) -> int:
    """counts words by splitting on whitespace and common punctuation/symbols that LLM tokenizers typically break apart"""
    if not text:
        return 0
    normalized_text = re.sub(r'[/:\-\s]+', ' ', text) # replace slashes, colons, hyphens, and whitespace with a single space
    words = normalized_text.split() # split by spaces to get the individual words
    return len(words)

def get_precise_token_count(base_url: str, model_name: str, system_content: str, user_content: str) -> int:
    """calculates the exact number of tokens as the model sees them, including the system prompt and user prompt"""
    if not base_url or not model_name:
        print(f"Error: Problem with LLM; Base URL: {base_url}; Model name: {model_name}", file=sys.stderr)
        sys.exit(1)
    try:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content}
        ]
        response = requests.post(
            f"{base_url}/tokenize",
            json={
                "model": model_name,
                "messages": messages,
                "add_generation_prompt": True,
            },
            timeout=30,
        )
        response.raise_for_status()
    except ConnectionError:
        print(f"Error: Could not connect to the LLM service at '{base_url}'. Is the server running?", file=sys.stderr)
        sys.exit(1)
    except HTTPError as e:
        print(f"HTTP Error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    return response.json()["count"]

def get_version(mod_name):
    """returns version of python package/module"""
    try:
        # import dynamically using just the name you typed
        mod = importlib.import_module(mod_name)
    except ImportError:
        return "Not installed"
    # check if the module exposes a version attribute directly
    if hasattr(mod, "__version__"):
        return mod.__version__
    # automatically query Python's built-in registry to find the PyPI package name
    try:
        dist_names = importlib.metadata.packages_distributions().get(mod_name, [])
        for dist in dist_names:
            try:
                return importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:
                continue
    except Exception:
        pass
    # fallback to checking metadata under the import name itself
    try:
        return importlib.metadata.version(mod_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

def print_versions(modules: []):
    print("Versions:")
    for name in modules:
        print(f"- {name} == {get_version(name)}")
