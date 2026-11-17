# -*- coding: utf-8 -*-
# About: Checks of the files in data/. They are a part of the project as much as the code:
#        the scripts read them, and the benchmark scores are only as good as the surveys.
#
# The files are edited by hand and produced by LLMs, so small damage is easy: a missing field, a survey row
# with a wrong number of columns, an answer listed twice, a compressed text pointing to a dialogue that was renamed.
# The scripts would not crash on most of these. They would skip things quietly or score a question unfairly.
# These checks turn such quiet damage into a failed test that names the file and the place.
import tomllib
import pytest
import useful_utils as uu
import benchmark_compressed
from conftest import DATA_DIR

def load(path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)

# The lists are built when pytest collects the tests, so every file gets its own, separately reported test.
ALL_FILES        = sorted(DATA_DIR.glob("*.toml"))
DATA_FILES       = sorted(DATA_DIR.glob("data.*.toml"))
COMPRESSED_FILES = sorted(DATA_DIR.glob("compress.*.toml"))
PROMPT_FILES     = sorted(DATA_DIR.glob("prompts.*.toml"))

def file_name(path) -> str:
    return path.name

def all_data_entries() -> dict:
    """{data_id: entry} for every [[data]] entry of every data file"""
    return {d["data_id"]: d for path in DATA_FILES for d in load(path).get("data", [])}

def test_data_directory_was_found():
    """guards the checks below: with a wrong path they would have no files and pass without checking anything"""
    assert DATA_FILES and COMPRESSED_FILES and PROMPT_FILES

@pytest.mark.parametrize("path", ALL_FILES, ids=file_name)
def test_file_is_valid_toml(path):
    assert load(path)

# ---- data files: texts with surveys ----------------------------------------------------------------------------------

@pytest.mark.parametrize("path", DATA_FILES, ids=file_name)
def test_data_entries_have_the_fields_the_scripts_read(path):
    entries = load(path).get("data", [])
    assert entries, "no [[data]] entry"
    for d in entries:
        assert d.get("data_id"), "an entry without 'data_id'"
        assert d.get("type"), f"{d['data_id']}: no 'type'"
        assert isinstance(d.get("text"), str) and d["text"].strip(), f"{d['data_id']}: no 'text'"

def test_data_ids_are_unique_across_all_data_files():
    """compressed texts and scores refer to a text by its 'data_id' alone"""
    ids = [d["data_id"] for path in DATA_FILES for d in load(path).get("data", [])]
    assert sorted(ids) == sorted(set(ids))

@pytest.mark.parametrize("path", DATA_FILES, ids=file_name)
def test_word_counts_match_the_text(path):
    """a text edited by hand keeps its old 'text_words' until estimate_tokens.py is run again"""
    for d in load(path).get("data", []):
        if "text_words" in d:
            assert d["text_words"] == uu.count_words(d["text"]), f"{d['data_id']}: stale 'text_words', run utils/estimate_tokens.py"

@pytest.mark.parametrize("path", DATA_FILES, ids=file_name)
def test_survey_rows_are_complete_and_fair(path):
    """a row is: [question, correct answer, "wrong | wrong | ...", evidence]"""
    for d in load(path).get("data", []):
        survey = d.get("single_choice_survey", [])
        assert survey, f"{d['data_id']}: no survey"
        questions = [row[0] for row in survey]
        assert len(questions) == len(set(questions)), f"{d['data_id']}: a question is asked twice"
        for number, row in enumerate(survey, start=1):
            where = f"{d['data_id']}, question {number}"
            assert len(row) == 4, f"{where}: expected 4 columns, found {len(row)}"
            assert all(isinstance(column, str) and column.strip() for column in row), f"{where}: an empty column"
            question, correct, wrong, evidence = row
            # '|' separates the wrong answers, so it cannot be used inside an answer
            assert "|" not in correct, f"{where}: '|' in the correct answer"
            wrong_answers = [w.strip() for w in wrong.split("|")]
            assert all(wrong_answers), f"{where}: an empty wrong answer"
            answers = [correct.strip()] + wrong_answers
            # a question needs something to choose from, and every answer needs a letter
            assert 2 <= len(answers) <= len(benchmark_compressed.LETTERS), f"{where}: {len(answers)} answers"
            # the same answer twice would be right under one letter and wrong under another
            assert len(answers) == len(set(answers)), f"{where}: an answer is listed twice"

# ---- compressed files ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", COMPRESSED_FILES, ids=file_name)
def test_compressed_results_have_the_fields_the_benchmark_reads(path):
    results = load(path).get("results", [])
    assert results, "no [[results]] entry"
    for number, r in enumerate(results, start=1):
        where = f"result {number}"
        for field in ("data_id", "prompt_id", "source_llm"):
            assert isinstance(r.get(field), str) and r[field], f"{where}: no '{field}'"
        assert isinstance(r.get("target_words"), int), f"{where}: 'target_words' is not a number"
        assert isinstance(r.get("text"), str) and r["text"].strip(), f"{where}: no 'text'"

@pytest.mark.parametrize("path", COMPRESSED_FILES, ids=file_name)
def test_compressed_results_point_to_an_existing_text(path):
    """a result with an unknown 'data_id' is skipped by the benchmark without an error"""
    known_ids = set(all_data_entries())
    unknown = {r["data_id"] for r in load(path)["results"]} - known_ids
    assert not unknown, f"results for unknown data_id: {sorted(unknown)}"

@pytest.mark.parametrize("path", COMPRESSED_FILES, ids=file_name)
def test_compressed_results_are_not_duplicated(path):
    """the same text, prompt, LLM and target length twice means a file was merged or generated twice"""
    keys = [(r["data_id"], r["prompt_id"], r["source_llm"], r["target_words"]) for r in load(path)["results"]]
    assert len(keys) == len(set(keys))

@pytest.mark.parametrize("path", COMPRESSED_FILES, ids=file_name)
def test_compressed_word_counts_match_the_text(path):
    for number, r in enumerate(load(path)["results"], start=1):
        if "text_words" in r:
            assert r["text_words"] == uu.count_words(r["text"]), f"result {number}: stale 'text_words', run utils/estimate_tokens.py"

# ---- prompt files ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", PROMPT_FILES, ids=file_name)
def test_compress_prompts_are_usable(path):
    prompts = load(path).get("prompts", [])
    assert prompts, "no [[prompts]] entry"
    ids = [p.get("prompt_id") for p in prompts]
    assert all(ids) and len(ids) == len(set(ids)), "prompt ids must be present and unique"
    for p in prompts:
        assert p.get("type"), f"{p['prompt_id']}: no 'type'"
        # both templates must exist (one may be empty), compress_Gemma26B.py skips a prompt without them
        assert "system_prompt_template" in p and "user_prompt_template" in p, f"{p['prompt_id']}: a template is missing"
        both = p["system_prompt_template"] + p["user_prompt_template"]
        # without the tags the LLM would get no dialogue to compress, or no target length
        assert "{dialogue}" in both, f"{p['prompt_id']}: no {{dialogue}} tag"
        assert "{target_words}" in both, f"{p['prompt_id']}: no {{target_words}} tag"
