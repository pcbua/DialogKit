# -*- coding: utf-8 -*-
# About: Tests of utils/estimate_tokens.py, the script that adds word and token counts to a TOML file.
import pytest
import useful_utils as uu
import estimate_tokens as script

@pytest.fixture
def input_document() -> dict:
    """a file with both sections the script knows ([[data]], [[results]]) and things it must leave alone"""
    return {
        "configs": {"Some_LLM": {"TEMPERATURE": 0.3}},
        "data": [
            {"data_id": "t001", "text": "Agent: Hello there", "speakers": ["Agent"]},
            {"data_id": "t002", "note": "an entry without text"},
        ],
        "results": [
            {"data_id": "t001", "target_words": 100, "text": "short-text", "text_words": 999, "text_tokens": 999},
        ],
    }

def test_adds_counts_to_every_text_and_keeps_the_rest(write_toml, read_toml, run_main, tmp_path, input_document, fake_token_count):
    fake_token_count.count = lambda system_text, user_text: len(user_text) # a made up but predictable count
    input_path, output_path = write_toml("in.toml", input_document), tmp_path / "out.toml"

    run_main(script, input_path, output_path)
    output = read_toml(output_path)

    # counts are added to texts in [[data]] ...
    assert output["data"][0]["text_words"]  == 3
    assert output["data"][0]["text_tokens"] == len("Agent: Hello there")
    # ... and old counts in [[results]] are replaced with fresh ones
    assert output["results"][0]["text_words"]  == 2
    assert output["results"][0]["text_tokens"] == len("short-text")
    # an entry without text gets no counts
    assert output["data"][1] == {"data_id": "t002", "note": "an entry without text"}
    # everything else is copied as it was
    assert output["configs"] == input_document["configs"]
    assert output["data"][0]["speakers"] == ["Agent"]
    assert output["results"][0]["target_words"] == 100

def test_counts_tokens_of_the_text_alone(write_toml, run_main, tmp_path, input_document, fake_token_count):
    """the text goes as the user message with an empty system message, to the model set in the script"""
    run_main(script, write_toml("in.toml", input_document), tmp_path / "out.toml")
    assert fake_token_count.calls == [
        (script.BASE_URL, script.MODEL_NAME, "", "Agent: Hello there"),
        (script.BASE_URL, script.MODEL_NAME, "", "short-text"),
    ]

def test_creates_missing_output_directory(write_toml, run_main, tmp_path, input_document, fake_token_count):
    output_path = tmp_path / "new" / "dir" / "out.toml"
    run_main(script, write_toml("in.toml", input_document), output_path)
    assert output_path.is_file()

def test_refuses_to_overwrite_an_existing_output(write_toml, run_main, input_document, fake_token_count, capsys):
    """results of an LLM run take long to make, so the script never writes over an existing file"""
    input_path  = write_toml("in.toml", input_document)
    output_path = write_toml("out.toml", {"precious": True})
    with pytest.raises(SystemExit) as stop:
        run_main(script, input_path, output_path)
    assert stop.value.code == 1
    assert "already exists" in capsys.readouterr().err
    assert fake_token_count.calls == [] # stopped before any work

def test_stops_when_the_input_is_missing(run_main, tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        run_main(script, tmp_path / "missing.toml", tmp_path / "out.toml")
    assert stop.value.code == 1
    assert "does not exist" in capsys.readouterr().err

def test_stops_when_the_input_is_a_directory(run_main, tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        run_main(script, tmp_path, tmp_path / "out.toml")
    assert stop.value.code == 1
    assert "not a valid file" in capsys.readouterr().err

def test_stops_without_output_when_the_input_is_not_toml(run_main, tmp_path, capsys):
    input_path = tmp_path / "broken.toml"
    input_path.write_text("this is [not valid TOML")
    with pytest.raises(SystemExit) as stop:
        run_main(script, input_path, tmp_path / "out.toml")
    assert stop.value.code == 1
    assert not (tmp_path / "out.toml").exists()
