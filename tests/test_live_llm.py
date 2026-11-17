# -*- coding: utf-8 -*-
# About: Smoke tests against a real LLM server. They are NOT run by default.
# How to use it (start the LLM server first, see docs/index.md):
#   uv run pytest -m live
#
# All other tests use fakes, so they prove that our code is right, but not that the real server still
# understands us. These few tests close that gap: they send real queries, the same way the scripts do.
# Run them after changing the server, the model or the LLM settings in the scripts.
# They check only things any sane model gets right, so a failure means a broken connection, not a bad day of the LLM.
import asyncio
import pytest
import requests
import useful_utils as uu
import benchmark_compressed

pytestmark = pytest.mark.live # marks every test in this file

BASE_URL   = benchmark_compressed.LLM_CONFIG["BASE_URL"]
MODEL_NAME = benchmark_compressed.LLM_CONFIG["MODEL_NAME"]

@pytest.fixture(scope="module", autouse=True)
def llm_server():
    """fails all tests of this file with one clear reason when the server is not there"""
    try:
        response = requests.get(f"{BASE_URL}/v1/models", timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        pytest.fail(f"The LLM server at {BASE_URL} is not reachable: {e}")
    return response.json()

def test_server_serves_the_model_set_in_the_scripts(llm_server):
    served = [model["id"] for model in llm_server["data"]]
    assert MODEL_NAME in served

def test_token_count_of_a_short_text():
    count = uu.get_precise_token_count(BASE_URL, MODEL_NAME, "", "Hello, how are you today?")
    assert 5 <= count <= 60 # 7 tokens of text plus the chat wrapping

def test_reader_answers_an_obvious_question_with_the_right_letter():
    """the full path of the benchmark for one question: chat with history, short answer, letter parsing"""
    from openai import AsyncOpenAI
    question = {"question": "What colour is Anna's car?", "answers": ["Blue.", "Green.", "Red.", "White.", "Black."], "correct_letter": "C"}
    async def ask():
        client = AsyncOpenAI(base_url=f"{BASE_URL}/v1", api_key=benchmark_compressed.LLM_CONFIG["API_KEY"])
        return await benchmark_compressed.ask_question(asyncio.Semaphore(1), client, "Anna bought a red car last week.", question)
    assert asyncio.run(ask()) == "C"
