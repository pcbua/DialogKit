# -*- coding: utf-8 -*-
# About: Tests of utils/benchmark_replies.py, the script that repeats the benchmark queries and keeps the raw replies.
#
# The replies are only worth something if they belong to the scores in the benchmark files, so the main promise
# tested here is: this script asks exactly the queries the benchmark asks (same texts, questions and order of answers).
import re
import json
import pytest
import benchmark_compressed as bench
import benchmark_replies as script
from test_benchmark_compressed import split_query, FACTS, CORRECT_ANSWERS # the same fake survey as the benchmark tests
from fakes import FakeLLM, FakeResponse

@pytest.fixture
def files(write_toml, sample_data):
    """a data file with one dialogue and two compressed files (two benchmark runs)"""
    data = {"data": [sample_data["data"][0]]}
    good = {"results": [
        {"data_id": "t001", "prompt_id": "p01", "source_llm": "Good LLM", "target_words": 200, "text_tokens": 12, "text": "Marta calls. Julia is the daughter. Weight: 85 kilos."},
        {"data_id": "t999", "prompt_id": "p01", "source_llm": "Good LLM", "target_words": 100, "text": "A version of another dialogue."},
    ]}
    poor = {"results": [
        {"data_id": "t001", "prompt_id": "p02", "source_llm": "Poor LLM", "target_words": 100, "text_tokens": 6, "text": "A phone call about health."},
    ]}
    return write_toml("data.t001.toml", data), write_toml("compress.Good.toml", good), write_toml("compress.Poor.toml", poor)

def reader(query: dict) -> str:
    """answers right when the fact is in the text; otherwise refuses (short budget) or names a letter late (long budget)"""
    given_text, question, answers = split_query(query)
    if FACTS[question] in given_text:
        return next(letter for letter, answer in answers.items() if answer == CORRECT_ANSWERS[question])
    return "The provided text does not contain this." if query["max_tokens"] <= 8 else "The text does not say. The answer is C"

@pytest.fixture
def server(monkeypatch):
    """the LLM server check passes, and both scripts get the same fake reader"""
    monkeypatch.setattr(bench.requests, "get", lambda url, timeout: FakeResponse({}))
    fake = FakeLLM(reply=reader)
    monkeypatch.setattr(bench, "AsyncOpenAI", fake.as_client_class)
    monkeypatch.setattr(script, "AsyncOpenAI", fake.as_client_class)
    return fake

def queries_of(calls: list) -> list:
    """what was asked, in a form that can be compared: (text, last message), sorted"""
    return sorted((c["messages"][1]["content"] if len(c["messages"]) == 4 else "", c["messages"][-1]["content"]) for c in calls)

def test_run_name_is_the_compressed_file_name_without_the_prefix():
    assert script.run_name("data/compress.Frontier.p02.d001.toml") == "Frontier.p02.d001"
    assert script.run_name("other.toml") == "other"

def test_queries_are_exactly_those_of_the_benchmark_for_the_same_seed(files, run_main, server):
    data_path, good, poor = files
    expected = list()
    for compressed in (good, poor): # the benchmark: one run per compressed file
        server.calls.clear()
        run_main(bench, data_path, compressed, "--seed", 7)
        expected += queries_of(server.calls)
    planned = script.plan_queries(bench.load_toml(data_path), {"Good": bench.load_toml(good), "Poor": bench.load_toml(poor)}, 7)
    built = sorted((q["text"] and bench.build_messages(q["text"], q["question"])[1]["content"] or "",
                    bench.build_messages(q["text"], q["question"])[-1]["content"]) for q in planned)
    assert built == sorted(expected)

def test_a_different_seed_gives_a_different_order_of_answers(files):
    data_path, good, _ = files
    plan = lambda seed: [q["question"]["answers"] for q in script.plan_queries(bench.load_toml(data_path), {"Good": bench.load_toml(good)}, seed)]
    assert plan(1) != plan(2)
    assert plan(1) == plan(1)

def test_plan_puts_the_reference_points_first_and_numbers_the_texts_like_the_benchmark_results(files):
    data_path, good, _ = files
    planned = script.plan_queries(bench.load_toml(data_path), {"Good": bench.load_toml(good)}, 42)
    texts = [(q["vidx"], q["type"]) for q in planned if q["qidx"] == 0]
    assert texts == [(0, "No Text"), (1, "Original Text"), (2, "Compressed Dialogue")] # the version of t999 is not asked

def test_script_saves_replies_with_parsed_letters_and_asks_letterless_ones_again(files, run_main, server, tmp_path):
    data_path, good, poor = files
    output = tmp_path / "replies.json"
    run_main(script, data_path, good, poor, "--output", output, "--long-sample", 2, "--long-tokens", 50)
    saved = json.loads(output.read_text(encoding="utf-8"))
    rows = saved["rows"]
    assert len(rows) == (3 + 3) * 3 # two runs, each: 2 reference points + its versions of t001, 3 questions
    assert {r["run"] for r in rows} == {"Good", "Poor"}
    assert all(set(r) >= {"reply", "parsed", "expected", "answers", "question", "vidx", "qidx"} for r in rows)
    original = [r for r in rows if r["type"] == "Original Text"]
    assert all(r["parsed"] == r["expected"] for r in original) # the honest fake finds every fact in the original
    assert any(r["parsed"] is None for r in rows)
    assert "text" not in rows[0] # the long texts are in the data files already
    assert len(saved["long"]) == 2
    assert all(r["long_reply"].endswith("The answer is C") for r in saved["long"]) # asked with the larger budget
    assert saved["seed"] == 42 and saved["long_tokens"] == 50

def test_script_refuses_to_overwrite_an_existing_output(files, run_main, server, write_toml, capsys):
    data_path, good, _ = files
    with pytest.raises(SystemExit):
        run_main(script, data_path, good, "--output", write_toml("existing.toml", {"precious": True}))
    assert "already exists" in capsys.readouterr().err

def test_script_uses_the_given_server_address(files, run_main, server, tmp_path, monkeypatch):
    data_path, good, _ = files
    monkeypatch.setitem(bench.LLM_CONFIG, "BASE_URL", bench.LLM_CONFIG["BASE_URL"]) # restored after the test
    run_main(script, data_path, good, "--output", tmp_path / "r.json", "--base-url", "http://10.0.0.5:8000/", "--long-sample", 0)
    assert server.client_kwargs["base_url"] == "http://10.0.0.5:8000/v1"
