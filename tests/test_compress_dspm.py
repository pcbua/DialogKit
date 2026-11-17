# -*- coding: utf-8 -*-
# About: Tests of utils/compress_DSPM.py, the script that compresses dialogues with the DSPM chat memory.
import asyncio
import pytest
import compress_DSPM as script

class FakeDSPMMemory:
    """stand-in for dspm.AsyncDSPMMemory: remembers the turns and returns a made up 'compressed' text"""
    created = [] # every memory made during a test, so a test can look inside

    def __init__(self, budget, llm_client, model):
        self.budget, self.llm_client, self.model = budget, llm_client, model
        self.turns, self.query = [], None
        FakeDSPMMemory.created.append(self)

    async def async_add_turn(self, role, text):
        self.turns.append((role, text))

    async def async_get_context(self, query):
        self.query = query
        return f"[DEC] context from {len(self.turns)} turns within {self.budget} tokens"

@pytest.fixture(autouse=True)
def fake_dspm(monkeypatch):
    """every test of this file runs with the fake DSPM memory instead of the real package"""
    FakeDSPMMemory.created = []
    monkeypatch.setattr(script, "AsyncDSPMMemory", FakeDSPMMemory)
    return FakeDSPMMemory

# ---- feeding a dialogue to DSPM --------------------------------------------------------------------------------------

def test_dialogue_is_fed_turn_by_turn_with_two_chat_roles(sample_data):
    """the Agent becomes the 'assistant', every other speaker the 'user'; the speaker name stays in the text"""
    memory = FakeDSPMMemory(budget=100, llm_client=None, model="m")
    result = asyncio.run(script.process_with_dspm(memory, sample_data["data"][0]["text"]))

    assert memory.turns == [
        ("assistant", "Agent:  Hello, this is Marta. Is this Edward?"),
        ("user",      "Caregiver:  No, it's Julia, his daughter."),
        ("user",      "Patient:  I'm here. My weight is 85 kilos."),
        ("assistant", "agent:  Thank you, Edward."), # the speaker name is matched without regard to letter case
    ]
    assert result == "[DEC] context from 4 turns within 100 tokens"
    assert memory.query # DSPM was asked for the context with some question

def test_lines_that_are_not_turns_are_left_out():
    """a turn is a line with 'Speaker: text'; empty lines and lines without a colon are not sent to DSPM"""
    memory = FakeDSPMMemory(budget=100, llm_client=None, model="m")
    asyncio.run(script.process_with_dspm(memory, "\n(phone rings)\nAgent: Hello\n\n"))
    assert memory.turns == [("assistant", "Agent: Hello")]

# ---- one compression -------------------------------------------------------------------------------------------------

def test_token_budget_is_derived_from_the_word_target(fake_llm):
    """DSPM takes a budget in tokens, the script works in words: 1 word is counted as 1.625 tokens"""
    result = asyncio.run(script.process_with_llm(asyncio.Semaphore(1), fake_llm, "my-model", "Agent: Hello", 400))
    memory = FakeDSPMMemory.created[0]
    assert memory.budget == 650
    assert (memory.llm_client, memory.model) == (fake_llm, "my-model")
    assert result == "[DEC] context from 1 turns within 650 tokens"

def test_failed_compression_is_reported_and_gives_no_result(fake_llm, monkeypatch, capsys):
    async def broken(memory, text):
        raise RuntimeError("dspm exploded")
    monkeypatch.setattr(script, "process_with_dspm", broken)
    assert asyncio.run(script.process_with_llm(asyncio.Semaphore(1), fake_llm, "my-model", "Agent: Hello", 400)) is None
    assert "dspm exploded" in capsys.readouterr().err

# ---- the whole script ------------------------------------------------------------------------------------------------

def test_script_compresses_every_dialogue_to_every_target_length(write_toml, sample_data, run_main, read_toml, tmp_path, monkeypatch, fake_llm):
    monkeypatch.setattr(script, "AsyncOpenAI", fake_llm.as_client_class)
    data_path = write_toml("data.t001.toml", sample_data)

    run_main(script, data_path, tmp_path / "compress.toml")
    output = read_toml(tmp_path / "compress.toml")

    # one result per target length for the dialogue; the story in the data file is not compressed
    results = output["results"]
    assert sorted(r["target_words"] for r in results) == sorted(script.TARGET_WORDS)
    for r in results:
        assert r["type"] == "Compressed Dialogue"
        assert r["data_id"] == "t001"
        assert r["source_llm"] == "DSPM/Gemma26B" # tells DSPM results apart from plain LLM results
        assert r["prompt_id"] == "NA"             # DSPM uses no prompt of ours
        assert r["text"] == f"[DEC] context from 4 turns within {round(r['target_words'] * 1.625)} tokens"
    assert output["configs"] == script.LLM_CONFIGS

def test_script_refuses_to_overwrite_an_existing_output(write_toml, sample_data, run_main, capsys):
    with pytest.raises(SystemExit) as stop:
        run_main(script, write_toml("data.t001.toml", sample_data), write_toml("existing.toml", {"precious": True}))
    assert stop.value.code == 1
    assert "already exists" in capsys.readouterr().err
    assert FakeDSPMMemory.created == []

def test_script_does_nothing_when_no_data_file_matches(run_main, tmp_path, capsys):
    run_main(script, tmp_path / "nothing-here.*.toml", tmp_path / "out.toml")
    assert "No files matched" in capsys.readouterr().out
    assert not (tmp_path / "out.toml").exists()
