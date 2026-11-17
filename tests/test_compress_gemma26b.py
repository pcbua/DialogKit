# -*- coding: utf-8 -*-
# About: Tests of utils/compress_Gemma26B.py, the script that compresses dialogues with prompts and a local LLM.
import asyncio
import pytest
import compress_Gemma26B as script

LLM_CONFIG = script.LLM_CONFIGS["Gemma26B_ThinkOff"]
PROMPT = {
    "type": "Compress Dialogue",
    "prompt_id": "p01",
    "system_prompt_template": "Shorten to {target_words} words.",
    "user_prompt_template": "Dialogue:\n{dialogue}",
}

def compress(fake_llm, prompt=PROMPT, text="Agent: Hello", n_words=200):
    """runs the single-query function of the script with a fake LLM"""
    return asyncio.run(script.process_with_llm(asyncio.Semaphore(1), fake_llm, LLM_CONFIG, prompt, text, n_words))

# ---- building blocks -------------------------------------------------------------------------------------------------

def test_prepare_prompt_fills_both_tags():
    template = "Shorten to {target_words} words: {dialogue} (about {target_words} words)"
    assert script.prepare_prompt(template, "Agent: Hi", 300) == "Shorten to 300 words: Agent: Hi (about 300 words)"

def test_iterate_inputs_takes_dialogues_and_skips_other_types(write_toml, sample_data):
    """stories are skipped on purpose: they may get a different kind of compression later"""
    path = write_toml("data.toml", sample_data)
    inputs = list(script.iterate_inputs([path]))
    assert [data_id for data_id, text in inputs] == ["t001"]
    assert inputs[0][1] == sample_data["data"][0]["text"]

def test_iterate_prompts_takes_only_complete_prompts_of_the_wanted_type():
    prompts = {"prompts": [
        PROMPT,
        {**PROMPT, "prompt_id": "other-type", "type": "Something Else"},
        {"type": "Compress Dialogue", "prompt_id": "no-templates"},
    ]}
    assert [p["prompt_id"] for p in script.iterate_prompts(prompts, "Compress Dialogue")] == ["p01"]

# ---- one LLM query ---------------------------------------------------------------------------------------------------

def test_query_sends_filled_prompts_and_returns_clean_text(fake_llm, fake_token_count):
    fake_llm.reply = "  The compressed dialogue.\n"
    assert compress(fake_llm, text="Agent: Hello", n_words=200) == "The compressed dialogue."

    query = fake_llm.calls[0]
    assert query["model"] == LLM_CONFIG["MODEL_NAME"]
    assert query["messages"] == [
        {"role": "system", "content": "Shorten to 200 words."},
        {"role": "user",   "content": "Dialogue:\nAgent: Hello"},
    ]
    assert query["temperature"] == LLM_CONFIG["TEMPERATURE"]
    assert query["extra_body"]["enable_reasoning"] is False # this config is the one without thinking

def test_query_gives_the_llm_all_the_room_left_in_the_context_window(fake_llm, fake_token_count):
    """answer budget = context window - prompt tokens - 1; a wrong budget cuts answers or gets queries rejected"""
    fake_token_count.count = 3000
    compress(fake_llm)
    assert fake_llm.calls[0]["max_tokens"] == LLM_CONFIG["MAX_MODEL_LEN"] - 3000 - 1
    # the count was made for exactly the prompts that were sent
    base_url, model_name, system_text, user_text = fake_token_count.calls[0]
    assert (system_text, user_text) == ("Shorten to 200 words.", "Dialogue:\nAgent: Hello")

def test_query_is_skipped_when_the_prompt_fills_the_context_window(fake_llm, fake_token_count, capsys):
    fake_token_count.count = LLM_CONFIG["MAX_MODEL_LEN"]
    assert compress(fake_llm) is None
    assert fake_llm.calls == [] # the LLM was not bothered
    assert "no room left" in capsys.readouterr().err

def test_empty_llm_answer_is_reported_and_gives_no_result(fake_llm, fake_token_count, capsys):
    """the LLM returns no text e.g. when all tokens went to thinking"""
    fake_llm.reply, fake_llm.finish_reason = None, "length"
    assert compress(fake_llm) is None
    assert "returned no text" in capsys.readouterr().err

def test_cut_off_answer_is_kept_with_a_warning(fake_llm, fake_token_count, capsys):
    fake_llm.reply, fake_llm.finish_reason = "A text cut in the", "length"
    assert compress(fake_llm) == "A text cut in the"
    assert "cut off" in capsys.readouterr().err

def test_failed_query_is_reported_and_gives_no_result(fake_llm, fake_token_count, capsys):
    """one failed query must not stop the run, the other queries may still succeed"""
    fake_llm.reply = RuntimeError("server exploded")
    assert compress(fake_llm) is None
    assert "server exploded" in capsys.readouterr().err

# ---- the whole script ------------------------------------------------------------------------------------------------

@pytest.fixture
def script_inputs(write_toml, sample_data):
    """prompts file and data file for a run of the whole script"""
    prompts = {"prompts": [PROMPT, {**PROMPT, "prompt_id": "p02"}]}
    return write_toml("prompts.toml", prompts), write_toml("data.t001.toml", sample_data)

def test_script_compresses_every_combination_and_saves_the_results(script_inputs, run_main, read_toml, tmp_path, monkeypatch, fake_llm, fake_token_count):
    monkeypatch.setattr(script, "AsyncOpenAI", fake_llm.as_client_class)
    fake_llm.reply = lambda query: f"compressed to {query['messages'][0]['content']}"
    prompts_path, data_path = script_inputs
    output_path = tmp_path / "out" / "compress.toml"

    run_main(script, prompts_path, tmp_path / "data.*.toml", output_path)
    output = read_toml(output_path)

    # one result for every LLM config x prompt x dialogue x target length; the story in the data file is not compressed
    results = output["results"]
    expected = {
        (llm, prompt_id, "t001", n_words)
        for llm in script.LLM_CONFIGS for prompt_id in ("p01", "p02") for n_words in script.TARGET_WORDS
    }
    assert {(r["source_llm"], r["prompt_id"], r["data_id"], r["target_words"]) for r in results} == expected
    assert len(results) == len(expected) # no duplicates
    # each result has the fields the other tools read
    for r in results:
        assert r["type"] == "Compressed Dialogue"
        assert r["text"] == f"compressed to Shorten to {r['target_words']} words."
        assert "date" in r
    # the LLM settings are saved with the results, so a file tells how it was made
    assert output["configs"] == script.LLM_CONFIGS
    assert fake_llm.client_kwargs["base_url"] == f"{LLM_CONFIG['BASE_URL']}/v1"

def test_script_saves_what_succeeded_when_some_queries_fail(script_inputs, run_main, read_toml, tmp_path, monkeypatch, fake_llm, fake_token_count):
    monkeypatch.setattr(script, "AsyncOpenAI", fake_llm.as_client_class)
    def reply(query): # the queries for 100 words fail
        return RuntimeError("fail") if query["messages"][0]["content"] == "Shorten to 100 words." else "ok"
    fake_llm.reply = reply
    prompts_path, data_path = script_inputs

    run_main(script, prompts_path, data_path, tmp_path / "compress.toml")
    results = read_toml(tmp_path / "compress.toml")["results"]

    assert results and all(r["target_words"] != 100 for r in results)
    assert len(results) == len(script.LLM_CONFIGS) * 2 * (len(script.TARGET_WORDS) - 1)

def test_script_refuses_to_overwrite_an_existing_output(script_inputs, write_toml, run_main, monkeypatch, fake_llm, fake_token_count, capsys):
    monkeypatch.setattr(script, "AsyncOpenAI", fake_llm.as_client_class)
    prompts_path, data_path = script_inputs
    with pytest.raises(SystemExit) as stop:
        run_main(script, prompts_path, data_path, write_toml("existing.toml", {"precious": True}))
    assert stop.value.code == 1
    assert "already exists" in capsys.readouterr().err
    assert fake_llm.calls == []

def test_script_stops_when_the_prompts_file_is_missing(run_main, tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        run_main(script, tmp_path / "missing.toml", tmp_path / "data.*.toml", tmp_path / "out.toml")
    assert stop.value.code == 1
    assert "does not exist" in capsys.readouterr().err

def test_script_does_nothing_when_no_data_file_matches(script_inputs, run_main, tmp_path, monkeypatch, fake_llm, capsys):
    monkeypatch.setattr(script, "AsyncOpenAI", fake_llm.as_client_class)
    prompts_path, data_path = script_inputs
    run_main(script, prompts_path, tmp_path / "nothing-here.*.toml", tmp_path / "out.toml")
    assert "No files matched" in capsys.readouterr().out
    assert not (tmp_path / "out.toml").exists()
