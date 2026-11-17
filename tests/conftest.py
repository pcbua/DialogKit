# -*- coding: utf-8 -*-
# About: Shared setup for all tests; pytest loads this file by itself before any test.
# How to use it:
#   uv run pytest                                  # run all tests (no LLM server needed)
#   uv run pytest tests/test_useful_utils.py       # run tests of one script
#   uv run pytest -k parse_letter                  # run tests with a matching name
#   uv run pytest -m live                          # run the few tests that need a real LLM server
#
# How the tests are organized:
#   - tests/test_<script>.py   tests of utils/<script>.py, one file per script
#   - tests/test_data_files.py checks of the files in data/ (they are a part of the project, too)
#   - tests/test_live_llm.py   smoke tests against a real LLM server, not run by default
#   - tests/fakes.py           stand-ins for the LLM, so the tests are fast, repeatable and need no GPU
#
# Two kinds of tests are used for every script:
#   - small tests of single functions (what does this function return for this input?),
#   - "end to end" tests that run the whole script, the same way as from the command line, on small
#     files in a temporary directory, with a fake LLM. They protect the things a user sees: files and messages.
import sys
import socket
import asyncio
import inspect
import tomllib
from pathlib import Path
import pytest
import tomli_w
import fakes

# must run before the test files import the scripts, see the function for the reason
PLACEHOLDER_PACKAGES = fakes.install_placeholders_for_missing_packages()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data"

def pytest_report_header(config):
    """adds a line to the top of the pytest report, so it is clear what the run was based on"""
    if PLACEHOLDER_PACKAGES:
        return f"DialogKit: packages not installed, replaced by placeholders: {', '.join(PLACEHOLDER_PACKAGES)}"
    return "DialogKit: all LLM packages are installed"

@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
    """blocks all network connections during a test (except for tests marked 'live')

    A test that reaches a real server is slow, and its result depends on that server and not only on our code.
    With this guard a forgotten fake fails loudly instead of quietly calling http://localhost:8000.
    """
    if request.node.get_closest_marker("live"):
        return
    def refuse(self, address):
        raise RuntimeError(f"A test tried to open a network connection to {address}. Use a fake from tests/fakes.py.")
    monkeypatch.setattr(socket.socket, "connect", refuse)

@pytest.fixture
def write_toml(tmp_path):
    """returns a function that saves a dict as a TOML file in a temporary directory and returns its path"""
    def _write(name: str, document: dict) -> Path:
        path = tmp_path / name
        with open(path, "wb") as f:
            tomli_w.dump(document, f, multiline_strings=True)
        return path
    return _write

@pytest.fixture
def read_toml():
    """returns a function that loads a TOML file"""
    def _read(path) -> dict:
        with open(path, "rb") as f:
            return tomllib.load(f)
    return _read

@pytest.fixture
def run_main(monkeypatch):
    """returns a function that runs a script like from the command line: run_main(script, "arg1", "arg2")

    The scripts read their arguments from sys.argv, so it is set for the time of the call.
    Works for both kinds of scripts: with a plain main() and with an async main().
    """
    def _run(script, *args):
        monkeypatch.setattr(sys, "argv", [script.__name__, *[str(a) for a in args]])
        result = script.main()
        if inspect.iscoroutine(result):
            result = asyncio.run(result)
        return result
    return _run

@pytest.fixture
def fake_llm():
    """a fake LLM client; set 'fake_llm.reply' in the test to control its answers"""
    return fakes.FakeLLM()

@pytest.fixture
def fake_token_count(monkeypatch):
    """replaces the token counting (a query to the LLM server) with a fake that always returns 100"""
    import useful_utils
    counter = fakes.FakeTokenCounter(count=100)
    monkeypatch.setattr(useful_utils, "get_precise_token_count", counter)
    return counter

# A small data file, in the same shape as data/data.d00*.toml: one dialogue, one story, each with a survey.
# The story is there on purpose: the compress scripts must skip it (stories may get their own compression later).
@pytest.fixture
def sample_data() -> dict:
    return {
        "data": [
            {
                "type": "Dialogue",
                "data_id": "t001",
                "text": "\nAgent:  Hello, this is Marta. Is this Edward?\nCaregiver:  No, it's Julia, his daughter.\nPatient:  I'm here. My weight is 85 kilos.\nagent:  Thank you, Edward.\n",
                "single_choice_survey": [
                    ["Who is Julia?", "Edward's daughter.", "Edward's wife. | Edward's nurse. | Edward's neighbor. | Edward's sister.", "Evidence: \"it's Julia, his daughter\""],
                    ["What is the Patient's weight?", "85 kilos.", "83 kilos. | 80 kilos. | 90 kilos. | 75 kilos.", "Evidence: \"My weight is 85 kilos.\""],
                    ["Who is calling?", "Marta.", "Julia. | Edward. | Anna. | Nobody.", "Evidence: \"this is Marta\""],
                ],
            },
            {
                "type": "Story",
                "data_id": "t002",
                "text": "Once upon a time: a fox met a crow.",
                "single_choice_survey": [
                    ["Who met the crow?", "A fox.", "A wolf. | A bear. | A cat. | A dog.", "Evidence: \"a fox met a crow\""],
                ],
            },
        ],
    }
