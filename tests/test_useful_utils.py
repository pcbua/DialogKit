# -*- coding: utf-8 -*-
# About: Tests of utils/useful_utils.py, the helpers shared by all scripts.
import sys
import types
import asyncio
import pytest
import requests
import useful_utils as uu
from fakes import FakeResponse

# ---- count_words -----------------------------------------------------------------------------------------------------
# The word counts are stored in the data files and compared with target budgets, so the counting rules must not drift.

@pytest.mark.parametrize("text, expected", [
    ("",                          0),
    (None,                        0),
    ("one",                       1),
    ("two words",                 2),
    ("  spaces   and\ttabs\nnewlines ", 4),
    ("well-known",                2), # a hyphen splits a word, like LLM tokenizers do
    ("and/or",                    2), # a slash, too
    ("Agent: Hello",              2), # the colon after a speaker name is not a word
    ("at 10:30",                  3), # a colon splits a time
    ("it's",                      1), # an apostrophe does not split
    ("- - -",                     0), # separators alone are not words
])
def test_count_words(text, expected):
    assert uu.count_words(text) == expected

# ---- time_it ---------------------------------------------------------------------------------------------------------

def test_time_it_keeps_result_and_name_of_a_plain_function(capsys):
    @uu.time_it
    def add(a, b):
        return a + b
    assert add(2, 3) == 5
    assert add.__name__ == "add"
    assert "[add] Execution time:" in capsys.readouterr().out

def test_time_it_keeps_result_of_an_async_function(capsys):
    @uu.time_it
    async def add(a, b):
        return a + b
    assert asyncio.run(add(2, 3)) == 5
    assert "[add] Execution time:" in capsys.readouterr().out

def test_time_it_reports_time_also_when_the_function_fails(capsys):
    @uu.time_it
    def broken():
        raise ValueError("boom")
    with pytest.raises(ValueError):
        broken()
    assert "[broken] Execution time:" in capsys.readouterr().out

@pytest.mark.parametrize("elapsed, expected", [
    (2.5,    "2.500 seconds"),
    (90,     "1.50 minutes"),
    (7200,   "2.00 hours"),
])
def test_time_it_picks_a_readable_unit(monkeypatch, capsys, elapsed, expected):
    clock = iter([1000.0, 1000.0 + elapsed]) # the two readings of the clock: start and end
    monkeypatch.setattr(uu.time, "perf_counter", lambda: next(clock))
    uu.time_it(lambda: None)()
    assert expected in capsys.readouterr().out

# ---- get_precise_token_count -----------------------------------------------------------------------------------------
# This function is the only place that asks the server for token counts. Its contract: return the count,
# or stop the script with exit code 1 and one short message. It must never leak a traceback or a wrong number.

def test_token_count_returns_the_count_and_asks_the_right_question(monkeypatch):
    sent = {}
    def fake_post(url, json, timeout):
        sent.update(url=url, json=json)
        return FakeResponse({"count": 42})
    monkeypatch.setattr(requests, "post", fake_post)

    assert uu.get_precise_token_count("http://llm:8000", "my-model", "system text", "user text") == 42
    assert sent["url"] == "http://llm:8000/tokenize"
    assert sent["json"]["model"] == "my-model"
    assert sent["json"]["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user",   "content": "user text"},
    ]
    # the count must include the tokens that open the LLM answer, they also take room in the context window
    assert sent["json"]["add_generation_prompt"] is True

@pytest.mark.parametrize("base_url, model_name", [(None, "my-model"), ("http://llm:8000", None), ("", "")])
def test_token_count_stops_without_server_address_or_model(base_url, model_name, capsys):
    with pytest.raises(SystemExit) as stop:
        uu.get_precise_token_count(base_url, model_name, "", "text")
    assert stop.value.code == 1
    assert "Error" in capsys.readouterr().err

def _http_error(status: int, text: str) -> requests.exceptions.HTTPError:
    response = requests.Response()
    response.status_code, response._content = status, text.encode()
    return requests.exceptions.HTTPError(response=response)

@pytest.mark.parametrize("post_result, expected_message", [
    (requests.exceptions.ConnectionError(),             "Could not connect"),
    (requests.exceptions.ReadTimeout(),                 "did not answer"),
    (FakeResponse(error=_http_error(500, "overloaded")), "HTTP Error 500: overloaded"),
    (FakeResponse({"something": "else"}),               "Unexpected answer"), # JSON without "count"
    (FakeResponse(None),                                "Unexpected answer"), # not JSON at all
], ids=["server down", "timeout", "http error", "no count in answer", "answer is not json"])
def test_token_count_stops_with_a_short_message_on_any_failure(monkeypatch, capsys, post_result, expected_message):
    def fake_post(*args, **kwargs):
        if isinstance(post_result, Exception):
            raise post_result
        return post_result
    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(SystemExit) as stop:
        uu.get_precise_token_count("http://llm:8000", "my-model", "", "text")
    assert stop.value.code == 1
    assert expected_message in capsys.readouterr().err

# ---- get_version / print_versions ------------------------------------------------------------------------------------

def test_get_version_reads_the_version_attribute(monkeypatch):
    module = types.ModuleType("made_up_module")
    module.__version__ = "1.2.3"
    monkeypatch.setitem(sys.modules, "made_up_module", module)
    assert uu.get_version("made_up_module") == "1.2.3"

def test_get_version_of_a_module_without_any_version(monkeypatch):
    monkeypatch.setitem(sys.modules, "made_up_module", types.ModuleType("made_up_module"))
    assert uu.get_version("made_up_module") == "unknown"

def test_get_version_of_a_missing_module():
    assert uu.get_version("surely_not_installed_module_xyz") == "Not installed"

def test_print_versions_prints_one_line_per_module(capsys):
    uu.print_versions(["requests", "surely_not_installed_module_xyz"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Versions:"
    assert lines[1].startswith("- requests == ")
    assert lines[2] == "- surely_not_installed_module_xyz == Not installed"
