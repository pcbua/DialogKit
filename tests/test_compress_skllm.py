# -*- coding: utf-8 -*-
# About: Tests of utils/compress_SkLLM.py, the script that compresses dialogues with the Scikit-LLM summarizer.
import asyncio
import threading
import pytest
import compress_SkLLM as script

CONCURRENCY = script.LLM_CONFIGS["Gemma26B"]["CONCURRENCY"]

class FakeSummarizer:
    """stand-in for skllm's GPTSummarizer: a blocking call that returns a made up summary"""
    created = []

    def __init__(self, model, max_words):
        self.model, self.max_words = model, max_words
        FakeSummarizer.created.append(self)

    def fit_transform(self, documents):
        return [f"summary of {len(documents)} document in {self.max_words} words"]

class FakeSKLLMConfig:
    """stand-in for skllm's SKLLMConfig: remembers the server settings"""
    url = key = None

    @classmethod
    def set_gpt_url(cls, url):
        cls.url = url

    @classmethod
    def set_openai_key(cls, key):
        cls.key = key

@pytest.fixture(autouse=True)
def fake_skllm(monkeypatch):
    """every test of this file runs with the fake Scikit-LLM classes instead of the real package"""
    FakeSummarizer.created = []
    monkeypatch.setattr(script, "GPTSummarizer", FakeSummarizer)
    monkeypatch.setattr(script, "SKLLMConfig", FakeSKLLMConfig)

def summarize(text="Agent: Hello", n_words=300):
    return asyncio.run(script.process_with_llm(asyncio.Semaphore(1), "my-model", text, n_words))

# ---- one compression -------------------------------------------------------------------------------------------------

def test_summarizer_gets_the_custom_server_model_and_the_word_limit():
    assert summarize(n_words=300) == "summary of 1 document in 300 words"
    summarizer = FakeSummarizer.created[0]
    assert summarizer.model == "custom_url::my-model" # the prefix tells Scikit-LLM to use our own server
    assert summarizer.max_words == 300

def test_no_summary_gives_no_result(monkeypatch):
    monkeypatch.setattr(FakeSummarizer, "fit_transform", lambda self, documents: [])
    assert summarize() is None

def test_failed_summary_is_reported_and_gives_no_result(monkeypatch, capsys):
    def broken(self, documents):
        raise RuntimeError("skllm exploded")
    monkeypatch.setattr(FakeSummarizer, "fit_transform", broken)
    assert summarize() is None
    assert "skllm exploded" in capsys.readouterr().err

# ---- the whole script ------------------------------------------------------------------------------------------------

def test_script_compresses_every_dialogue_to_every_target_length(write_toml, sample_data, run_main, read_toml, tmp_path):
    data_path = write_toml("data.t001.toml", sample_data)

    run_main(script, data_path, tmp_path / "compress.toml")
    output = read_toml(tmp_path / "compress.toml")

    # one result per target length for the dialogue; the story in the data file is not compressed
    results = output["results"]
    assert sorted(r["target_words"] for r in results) == sorted(script.TARGET_WORDS)
    for r in results:
        assert r["type"] == "Compressed Dialogue"
        assert r["data_id"] == "t001"
        assert r["source_llm"] == "Gemma26B"
        assert r["prompt_id"] == "NA" # Scikit-LLM uses its own prompt
        assert r["text"] == f"summary of 1 document in {r['target_words']} words"
    assert output["configs"] == script.LLM_CONFIGS
    # Scikit-LLM was pointed at our server
    assert FakeSKLLMConfig.url == f"{script.LLM_CONFIGS['Gemma26B']['BASE_URL']}/v1"
    assert FakeSKLLMConfig.key == script.LLM_CONFIGS["Gemma26B"]["API_KEY"]

def test_script_runs_as_many_queries_at_once_as_configured(write_toml, sample_data, run_main, tmp_path, monkeypatch):
    """the blocking summarizer runs in threads; there must be enough threads for CONCURRENCY queries at once

    By default asyncio has fewer threads (CPU cores + 4), which silently slowed the script down.
    The fake summarizer below does not return until CONCURRENCY calls are inside it at the same time.
    With too few threads that moment never comes, and the calls give up after the timeout.
    """
    lock, inside, everyone_arrived = threading.Lock(), [0], threading.Event()
    def wait_for_the_others(self, documents):
        with lock:
            inside[0] += 1
            if inside[0] >= CONCURRENCY:
                everyone_arrived.set()
        everyone_arrived.wait(timeout=2)
        with lock:
            inside[0] -= 1 # leaving; only calls that are inside at the same moment may add up to CONCURRENCY
        return ["summary"]
    monkeypatch.setattr(FakeSummarizer, "fit_transform", wait_for_the_others)

    # enough dialogues to have more queries than CONCURRENCY
    n_dialogues = CONCURRENCY // len(script.TARGET_WORDS) + 1
    dialogues = [{**sample_data["data"][0], "data_id": f"t{i:03}"} for i in range(n_dialogues)]

    run_main(script, write_toml("data.many.toml", {"data": dialogues}), tmp_path / "compress.toml")
    assert everyone_arrived.is_set(), f"fewer than {CONCURRENCY} queries were running at the same time"

def test_script_refuses_to_overwrite_an_existing_output(write_toml, sample_data, run_main, capsys):
    with pytest.raises(SystemExit) as stop:
        run_main(script, write_toml("data.t001.toml", sample_data), write_toml("existing.toml", {"precious": True}))
    assert stop.value.code == 1
    assert "already exists" in capsys.readouterr().err
    assert FakeSummarizer.created == []

def test_script_does_nothing_when_no_data_file_matches(run_main, tmp_path, capsys):
    run_main(script, tmp_path / "nothing-here.*.toml", tmp_path / "out.toml")
    assert "No files matched" in capsys.readouterr().out
    assert not (tmp_path / "out.toml").exists()
