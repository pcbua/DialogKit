# -*- coding: utf-8 -*-
# About: Tests of utils/benchmark_compressed.py, the script that scores compressed dialogues with a survey.
#
# The benchmark produces numbers that are used to compare compression methods, so a quiet mistake here
# (a wrong letter, a leaked answer, an unfair order) would spoil every conclusion drawn from them.
# That is why the rules of the benchmark are pinned down one by one below.
import re
import random
import asyncio
import pytest
import requests
import benchmark_compressed as script
from fakes import FakeLLM, FakeResponse

# ---- a fake "reader" LLM ---------------------------------------------------------------------------------------------

# What a reader must find in the text to answer each question of the sample survey (see 'sample_data' in conftest.py).
FACTS = {
    "Who is Julia?":                 "daughter",
    "What is the Patient's weight?": "85",
    "Who is calling?":               "Marta",
    "Who met the crow?":             "fox",
}
CORRECT_ANSWERS = {
    "Who is Julia?":                 "Edward's daughter.",
    "What is the Patient's weight?": "85 kilos.",
    "Who is calling?":               "Marta.",
    "Who met the crow?":             "A fox.",
}

def split_query(query: dict) -> tuple[str, str, dict]:
    """takes a chat query apart: (the text given as history, the question, {letter: answer})"""
    messages = query["messages"]
    given_text = messages[1]["content"] if len(messages) == 4 else ""
    last = messages[-1]["content"]
    question = re.search(r"Question: (.*)", last).group(1)
    answers = dict(re.findall(r"^([A-J])\) (.*)$", last, flags=re.MULTILINE))
    return given_text, question, answers

def honest_reader(query: dict) -> str:
    """a reader that answers right only when the needed fact is in the text it was given

    This mimics what the benchmark is about: a fact lost in compression cannot be answered.
    When the fact is missing, the reader picks a wrong letter.
    """
    given_text, question, answers = split_query(query)
    correct_letter = next(letter for letter, answer in answers.items() if answer == CORRECT_ANSWERS[question])
    if FACTS[question] in given_text:
        return correct_letter
    return next(letter for letter in answers if letter != correct_letter)

def numbered_survey(n: int) -> list:
    """a survey of n questions in the loaded form, with 1 correct and 4 wrong answers each"""
    return [{"question": f"Question {i}?", "correct": f"right {i}", "wrong": [f"wrong {i}{c}" for c in "abcd"]} for i in range(n)]

# ---- reading the survey ----------------------------------------------------------------------------------------------

def test_load_surveys_reads_questions_and_keeps_the_original_entries(sample_data):
    surveys, originals = script.load_surveys(sample_data)
    assert set(surveys) == {"t001", "t002"}
    assert surveys["t001"][0] == {
        "question": "Who is Julia?",
        "correct":  "Edward's daughter.",
        "wrong":    ["Edward's wife.", "Edward's nurse.", "Edward's neighbor.", "Edward's sister."],
    }
    assert originals["t001"]["text"] == sample_data["data"][0]["text"]

def test_load_surveys_skips_entries_without_a_survey_and_broken_rows(capsys):
    document = {"data": [
        {"data_id": "no-survey", "text": "text"},
        {"data_id": "broken-row", "text": "text", "single_choice_survey": [["only a question"], ["Q?", "right", "wrong 1 | wrong 2", "evidence"]]},
    ]}
    surveys, originals = script.load_surveys(document)
    assert set(surveys) == {"broken-row"}
    assert [q["question"] for q in surveys["broken-row"]] == ["Q?"]
    assert "malformed" in capsys.readouterr().err

# ---- the order of answers --------------------------------------------------------------------------------------------

def test_shuffle_keeps_all_answers_and_marks_the_correct_one():
    survey = numbered_survey(50)
    for original, shuffled in zip(survey, script.shuffle_survey(survey, random.Random(42))):
        assert shuffled["question"] == original["question"]
        assert sorted(shuffled["answers"]) == sorted([original["correct"]] + original["wrong"])
        position = script.LETTERS.index(shuffled["correct_letter"])
        assert shuffled["answers"][position] == original["correct"]

def test_shuffle_spreads_the_correct_answer_over_the_letters():
    """in the data file the correct answer is always first; if it stayed first, answering 'A' would score 100%"""
    letters = [q["correct_letter"] for q in script.shuffle_survey(numbered_survey(50), random.Random(42))]
    assert set(letters) == set("ABCDE")

def test_shuffle_is_repeatable_for_a_seed_and_differs_between_seeds():
    survey = numbered_survey(50)
    assert script.shuffle_survey(survey, random.Random(42)) == script.shuffle_survey(survey, random.Random(42))
    assert script.shuffle_survey(survey, random.Random(42)) != script.shuffle_survey(survey, random.Random(43))

def test_shuffle_draws_a_new_order_on_every_call():
    """one generator is used for a whole run, so every benchmarked text gets its own order of answers"""
    survey, rng = numbered_survey(50), random.Random(42)
    assert script.shuffle_survey(survey, rng) != script.shuffle_survey(survey, rng)

def test_shuffle_skips_a_question_with_more_answers_than_letters(capsys):
    survey = [{"question": "Q?", "correct": "right", "wrong": [f"wrong {i}" for i in range(len(script.LETTERS))]}]
    assert script.shuffle_survey(survey, random.Random(42)) == []
    assert "Too many answers" in capsys.readouterr().err

# ---- what gets benchmarked -------------------------------------------------------------------------------------------

def test_select_variants_takes_only_versions_of_the_surveyed_text(capsys):
    compressed = {"results": [
        {"data_id": "t001", "text": "first version"},
        {"data_id": "t001", "text": "second version", "type": "Compressed Dialogue"},
        {"data_id": "other", "text": "a version of another dialogue"},
        {"data_id": "t001", "text": ""},
        {"data_id": "t001"},
    ]}
    variants = script.select_variants(compressed, surveys={"t001": []})
    assert [v["text"] for v in variants] == ["first version", "second version"] # in file order, with or without 'type'
    assert "Skipped 3" in capsys.readouterr().out

def test_baselines_are_no_text_and_original_text(sample_data):
    surveys, originals = script.load_surveys(sample_data)
    baselines = script.make_baselines({"t001": originals["t001"]})
    assert [(b["type"], b["data_id"]) for b in baselines] == [("No Text", "t001"), ("Original Text", "t001")]
    assert baselines[0]["text"] == ""
    assert baselines[1]["text"] == sample_data["data"][0]["text"]

# ---- the chat sent to the LLM ----------------------------------------------------------------------------------------

QUESTION = {"question": "Who is Julia?", "answers": ["Edward's wife.", "Edward's daughter.", "Edward's nurse."], "correct_letter": "B"}

def test_messages_give_the_text_as_history_and_then_the_question():
    messages = script.build_messages("  Julia is the daughter.\n", QUESTION)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"].endswith("Julia is the daughter.")
    assert "Question: Who is Julia?" in messages[3]["content"]
    assert "A) Edward's wife.\nB) Edward's daughter.\nC) Edward's nurse." in messages[3]["content"]

def test_messages_without_a_text_have_no_history():
    """the 'No Text' baseline: the LLM must see the question alone, not an empty 'text'"""
    messages = script.build_messages("", QUESTION)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Question: Who is Julia?" in messages[1]["content"]

# The answers used in the parser tests: A) ... E), like in a real question.
ANSWERS = ["Edward's wife.", "Edward's daughter, Julia.", "A nurse from the care program.", "Edward's neighbor.", "Nobody answered."]

@pytest.mark.parametrize("reply, expected", [
    # 1. a bare letter, maybe decorated
    ("B",                         "B"),
    ("b",                         "B"),
    (" B \n",                     "B"),
    ("B)",                        "B"),
    ("(C)",                       "C"),
    ("**B**",                     "B"),
    ("\"D\"",                     "D"),
    # 2. a letter that opens the reply
    ("B) Edward's daughter",      "B"),
    ("b) Edward's daughter",      "B"),
    ("C. A nurse",                "C"),
    ("B: because Julia said so",  "B"),
    ("B, not C",                  "B"),
    ("D - the neighbor",          "D"),
    # 3. a letter announced in words
    ("Answer: B",                 "B"),  # the 'A' of 'Answer' is a part of a word, not a choice
    ("The answer is D.",          "D"),
    ("The correct answer is **C**", "C"),
    ("Option E",                  "E"),
    ("I am not sure, but B is correct", "B"),
    # 4. the text of an answer without a letter
    ("Edward's daughter, Julia.", "B"),
    ("edward's daughter julia",   "B"),  # letter case and punctuation do not matter
    ("A nurse from the care",     "C"),  # an answer cut off by the token limit; its first word is not the letter A
    ("Nobody answered.",          "E"),
    # 5. one capital letter somewhere in the reply
    ("I think B",                 "B"),
    ("Probably it's C!",          "C"),
], ids=repr)
def test_parse_letter_finds_the_choice(reply, expected):
    assert script.parse_letter(reply, ANSWERS) == expected

@pytest.mark.parametrize("reply", [
    "",
    "   ",
    None,
    "I do not know.",
    "42",
    "Edward",                        # letters inside a word are not a choice; too short to be the beginning of an answer
    "Edward's",                      # the beginning of three different answers
    "Nobody",                        # fits one answer only, but a single short word could fit by accident
    "F",                             # there is no answer F among 5 answers
    "B or C",                        # two choices are no choice
    "Either A or D.",
    "A or D",
    "A neighbor called, I believe.", # 'A' is a word here, and this is not the text of any answer
    "The answer is a nurse",         # a small 'a' in a sentence is a word, not a letter
    "it could be a or b",
], ids=repr)
def test_parse_letter_gives_no_answer_when_the_reply_is_not_clear(reply):
    """a guess would add random hits and misses to the scores, so an unclear reply counts as not answered"""
    assert script.parse_letter(reply, ANSWERS) is None

def test_parse_letter_accepts_only_as_many_letters_as_there_are_answers():
    assert script.parse_letter("C", ["one", "two", "three"]) == "C"
    assert script.parse_letter("D", ["one", "two", "three"]) is None

# ---- asking and scoring ----------------------------------------------------------------------------------------------

def ask(fake_llm, text="Julia is the daughter.", question=QUESTION):
    return asyncio.run(script.ask_question(asyncio.Semaphore(1), fake_llm, text, question))

def test_question_is_asked_with_settings_for_a_short_repeatable_answer(fake_llm):
    fake_llm.reply = "B"
    assert ask(fake_llm) == "B"
    query = fake_llm.calls[0]
    assert query["model"] == script.LLM_CONFIG["MODEL_NAME"]
    assert query["temperature"] == 0.0                         # the same answer on every run
    assert query["max_tokens"] <= 16                           # a letter, not an essay
    assert query["extra_body"]["enable_reasoning"] is False    # no thinking
    assert query["messages"] == script.build_messages("Julia is the daughter.", QUESTION)

def test_failed_query_counts_as_no_answer(fake_llm, capsys):
    fake_llm.reply = RuntimeError("server exploded")
    assert ask(fake_llm) is None
    assert "server exploded" in capsys.readouterr().err

def test_variant_score_counts_correct_wrong_and_missing_answers(fake_llm):
    survey = [
        {"question": "Q1?", "answers": ["a", "b"], "correct_letter": "A"},
        {"question": "Q2?", "answers": ["a", "b"], "correct_letter": "B"},
        {"question": "Q3?", "answers": ["a", "b"], "correct_letter": "A"},
        {"question": "Q4?", "answers": ["a", "b"], "correct_letter": "B"},
    ]
    replies = {"Q1?": "A", "Q2?": "A", "Q3?": "no idea", "Q4?": "B"} # right, wrong, no answer, right
    fake_llm.reply = lambda query: replies[split_query(query)[1]]
    variant = {"data_id": "t001", "prompt_id": "p01", "source_llm": "Some LLM", "target_words": 100, "text": "four words of text"}

    score = asyncio.run(script.benchmark_variant(asyncio.Semaphore(4), fake_llm, variant, survey))

    assert (score["questions"], score["correct"], score["no_answer"], score["accuracy"]) == (4, 2, 1, 0.5)
    assert score["wrong_questions"] == [2, 3] # numbered from 1; a missing answer is a wrong answer
    assert (score["data_id"], score["prompt_id"], score["source_llm"], score["target_words"]) == ("t001", "p01", "Some LLM", 100)
    assert score["type"] == "Compressed Dialogue"
    assert score["text_words"] == 4 # counted here, because the compressed file gave no count

# ---- the whole script ------------------------------------------------------------------------------------------------

@pytest.fixture
def llm_server_up(monkeypatch):
    """makes the 'is the LLM server running?' check pass"""
    monkeypatch.setattr(script.requests, "get", lambda url, timeout: FakeResponse({}))

@pytest.fixture
def reader(monkeypatch):
    """the script gets the honest fake reader instead of a real LLM"""
    fake_llm = FakeLLM(reply=honest_reader)
    monkeypatch.setattr(script, "AsyncOpenAI", fake_llm.as_client_class)
    return fake_llm

@pytest.fixture
def benchmark_files(write_toml, sample_data):
    """a data file (only the dialogue t001) and a compressed file with versions of different quality"""
    data = {"data": [sample_data["data"][0]]}
    compressed = {"results": [
        {"data_id": "t001", "prompt_id": "p01", "source_llm": "Good LLM", "target_words": 200, "text": "Marta calls. Julia is the daughter. Weight: 85 kilos."},
        {"data_id": "t001", "prompt_id": "p01", "source_llm": "Good LLM", "target_words": 100, "text": "Marta calls. Weight: 85 kilos."},
        {"data_id": "t001", "prompt_id": "p02", "source_llm": "Poor LLM", "target_words": 100, "text": "A phone call about health."},
        {"data_id": "t999", "prompt_id": "p01", "source_llm": "Good LLM", "target_words": 100, "text": "A version of another dialogue."},
    ]}
    return write_toml("data.t001.toml", data), write_toml("compress.toml", compressed)

def score_lines(output: str) -> list:
    """the table lines of the printed report, as lists of columns"""
    return [line.split() for line in output.splitlines() if line.startswith("t001")]

def test_script_prints_baselines_first_and_then_every_version_in_file_order(benchmark_files, run_main, llm_server_up, reader, capsys):
    run_main(script, *benchmark_files)
    lines = score_lines(capsys.readouterr().out)

    # the last three columns are: correct/questions, accuracy, no answer
    assert [line[-3:] for line in lines] == [
        ["0/3", "0.0%",   "0"], # no text:        nothing can be answered
        ["3/3", "100.0%", "0"], # original text:  everything can be answered
        ["3/3", "100.0%", "0"], # a version that kept all three facts
        ["2/3", "66.7%",  "0"], # a version that lost one fact
        ["0/3", "0.0%",   "0"], # a version that lost all facts
    ]
    assert "(no text)" in " ".join(lines[0]) and "(original text)" in " ".join(lines[1])
    # the version of another dialogue was not benchmarked: 5 texts x 3 questions
    assert len(reader.calls) == 15

def test_script_draws_a_new_order_of_answers_for_every_query_and_never_shows_the_evidence(benchmark_files, run_main, llm_server_up, reader):
    run_main(script, *benchmark_files)
    orders_seen = {}
    for query in reader.calls:
        given_text, question, answers = split_query(query)
        # whatever the order, it is always the same five answers of that question
        assert CORRECT_ANSWERS[question] in answers.values() and len(answers) == 5
        orders_seen.setdefault(question, set()).add(tuple(answers.values()))
        # the evidence column quotes the original text, showing it would give the answer away
        assert "Evidence" not in str(query["messages"])
    # each question was asked about 5 texts; one shared order would mean the answers are drawn once, not per query
    assert len(orders_seen) == 3
    assert all(len(orders) > 1 for orders in orders_seen.values())

def test_script_run_is_repeatable_with_the_same_seed(benchmark_files, run_main, llm_server_up, reader):
    """the same seed gives exactly the same queries, another seed gives other orders of answers"""
    def queries_of_a_run(seed):
        reader.calls.clear()
        run_main(script, *benchmark_files, "--seed", seed)
        return [query["messages"] for query in reader.calls]
    assert queries_of_a_run(1) == queries_of_a_run(1)
    assert queries_of_a_run(1) != queries_of_a_run(2)

def test_script_saves_detailed_scores_to_the_output_file(benchmark_files, run_main, read_toml, tmp_path, llm_server_up, reader):
    output_path = tmp_path / "out" / "benchmark.toml"
    run_main(script, *benchmark_files, "--output", output_path, "--seed", 7)
    output = read_toml(output_path)

    assert [(r["type"], r["accuracy"]) for r in output["results"]] == [
        ("No Text", 0.0), ("Original Text", 1.0), ("Compressed Dialogue", 1.0), ("Compressed Dialogue", 0.6667), ("Compressed Dialogue", 0.0),
    ]
    assert output["results"][3]["wrong_questions"] == [1] # the version without "daughter" fails the first question
    assert (output["results"][3]["source_llm"], output["results"][3]["target_words"]) == ("Good LLM", 100)
    # the file says how to repeat the run
    assert output["benchmark"]["seed"] == 7
    assert output["benchmark"]["data_path"] == str(benchmark_files[0])
    assert output["benchmark"]["compressed_path"] == str(benchmark_files[1])
    assert output["configs"] == {script.LLM_CONFIG_NAME: script.LLM_CONFIG}

def test_script_scores_do_not_depend_on_the_seed(benchmark_files, run_main, llm_server_up, reader, capsys):
    """the seed changes the letters, not the knowledge: an honest reader scores the same"""
    run_main(script, *benchmark_files, "--seed", 1)
    first = [line[-3:] for line in score_lines(capsys.readouterr().out)]
    run_main(script, *benchmark_files, "--seed", 2)
    second = [line[-3:] for line in score_lines(capsys.readouterr().out)]
    assert first == second

def test_script_stops_before_any_query_when_the_llm_server_is_down(benchmark_files, run_main, monkeypatch, reader, capsys):
    """otherwise every query would fail, and the report would look like a real, very bad result"""
    def refuse(url, timeout):
        raise requests.exceptions.ConnectionError("connection refused")
    monkeypatch.setattr(script.requests, "get", refuse)
    with pytest.raises(SystemExit) as stop:
        run_main(script, *benchmark_files)
    assert stop.value.code == 1
    assert "Could not reach the LLM service" in capsys.readouterr().err
    assert reader.calls == []

def test_script_stops_when_no_version_belongs_to_the_data_file(write_toml, sample_data, run_main, llm_server_up, reader, capsys):
    data_path = write_toml("data.toml", {"data": [sample_data["data"][0]]})
    compressed_path = write_toml("compress.toml", {"results": [{"data_id": "t999", "text": "another dialogue"}]})
    with pytest.raises(SystemExit) as stop:
        run_main(script, data_path, compressed_path)
    assert stop.value.code == 1
    assert "No compressed dialogue" in capsys.readouterr().err

def test_script_stops_when_the_data_file_has_no_survey(write_toml, run_main, llm_server_up, reader, capsys):
    data_path = write_toml("data.toml", {"data": [{"data_id": "t001", "text": "Agent: Hello"}]})
    compressed_path = write_toml("compress.toml", {"results": [{"data_id": "t001", "text": "Hello"}]})
    with pytest.raises(SystemExit) as stop:
        run_main(script, data_path, compressed_path)
    assert stop.value.code == 1
    assert "No 'single_choice_survey'" in capsys.readouterr().err

def test_script_refuses_to_overwrite_an_existing_output(benchmark_files, write_toml, run_main, llm_server_up, reader, capsys):
    with pytest.raises(SystemExit) as stop:
        run_main(script, *benchmark_files, "--output", write_toml("existing.toml", {"precious": True}))
    assert stop.value.code == 1
    assert "already exists" in capsys.readouterr().err
    assert reader.calls == []

def test_script_stops_when_an_input_file_is_missing(benchmark_files, run_main, tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        run_main(script, benchmark_files[0], tmp_path / "missing.toml")
    assert stop.value.code == 1
    assert "does not exist" in capsys.readouterr().err

# ---- the legend of the DSPM tags (--tag-legend) ------------------------------------------------------------------

def test_messages_with_tag_legend_put_the_text_between_the_legend_and_the_con_rule():
    q = {"question": "Who is Julia?", "answers": ["Edward's daughter.", "Edward's wife."], "correct_letter": "A"}
    history = script.build_messages("[DEC] Julia is the daughter.", q, tag_legend=True)[1]["content"]
    assert history.index(script.TAG_LEGEND) < history.index("[DEC] Julia is the daughter.") < history.index(script.TAG_RULE)
    assert "[CON] = a hard constraint that must be followed" in history
    # without the option nothing changes
    plain = script.build_messages("[DEC] Julia is the daughter.", q)[1]["content"]
    assert script.TAG_LEGEND not in plain and script.TAG_RULE not in plain

def test_script_with_tag_legend_gives_the_legend_only_with_compressed_texts(benchmark_files, run_main, llm_server_up, reader, read_toml, tmp_path):
    output_path = tmp_path / "scores.toml"
    run_main(script, *benchmark_files, "--tag-legend", "--output", output_path)
    with_text = [c for c in reader.calls if len(c["messages"]) == 4]
    original = [c for c in with_text if "Agent:  Hello, this is Marta" in c["messages"][1]["content"]]
    compressed = [c for c in with_text if c not in original]
    assert original and all(script.TAG_LEGEND not in c["messages"][1]["content"] for c in original)
    assert compressed and all(script.TAG_LEGEND in c["messages"][1]["content"] for c in compressed)
    assert read_toml(output_path)["benchmark"]["tag_legend"] is True

def test_script_without_tag_legend_records_that_in_the_output(benchmark_files, run_main, llm_server_up, reader, read_toml, tmp_path):
    output_path = tmp_path / "scores.toml"
    run_main(script, *benchmark_files, "--output", output_path)
    assert all(script.TAG_LEGEND not in m["content"] for c in reader.calls for m in c["messages"])
    assert read_toml(output_path)["benchmark"]["tag_legend"] is False
