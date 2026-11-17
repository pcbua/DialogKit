# -*- coding: utf-8 -*-
# About: Tests of utils/benchmark_report.py, the script that writes the Markdown report from the benchmark results.
#
# The report is read by people who draw conclusions from it, so the numbers in it must be the numbers in the files,
# and the comparisons must be fair (prompts at the same real length). Charts are not tested here: they need matplotlib,
# which is not a part of the test setup, and they are only a picture of the numbers that are tested.
import math
import json
import pytest
import benchmark_report as script

def result(data_id="t001", type_="Compressed Dialogue", prompt="p01", source="LLM", target=100, tokens=50, words=40, correct=5, questions=10, no_answer=0):
    return {"type": type_, "data_id": data_id, "prompt_id": prompt, "source_llm": source, "target_words": target, "text_words": words,
            "text_tokens": tokens, "questions": questions, "correct": correct, "no_answer": no_answer, "accuracy": round(correct / questions, 4), "wrong_questions": []}

def bench_file(compressed_path, results, data_path="data/data.t001.toml"):
    refs = [result(type_="No Text", prompt="-", source="(no text)", target=0, tokens=0, words=0, correct=1, no_answer=8),
            result(type_="Original Text", prompt="-", source="(original text)", target=0, tokens=100, words=80, correct=10)]
    return {"configs": {"Reader": {"MODEL_NAME": "reader-model", "TEMPERATURE": 0.0, "TOP_P": 1.0, "MAX_ANSWER_TOKENS": 8, "ENABLE_REASONING": False, "CONCURRENCY": 4}},
            "benchmark": {"type": "Benchmark Compressed Dialogue", "date": "2026-11-17", "data_path": data_path, "compressed_path": compressed_path,
                          "reader_llm": "Reader", "seed": 42},
            "results": refs + results}

@pytest.fixture
def result_files(write_toml):
    """two runs of one dialogue: a method with three prompts (p03 writes longer texts) and SkLLM"""
    prompts = []
    for p, extra in (("p01", 0), ("p02", 0), ("p03", 10)):
        for tokens in (10, 20, 40):
            tokens += extra
            correct = min(10, round(2 + 2 * math.log2(tokens / 10)) + (1 if p != "p01" else 0))
            prompts.append(result(prompt=p, tokens=tokens, target=tokens, words=tokens, correct=correct, no_answer=10 - correct))
    return [write_toml("benchmark.Gemma26B.t001.toml", bench_file("data/compress.Gemma26B.toml", prompts)),
            write_toml("benchmark.SkLLM.t001.toml", bench_file("data/compress.SkLLM.toml", [result(prompt="NA", tokens=8, correct=2, no_answer=7)]))]

def test_lsq_fit_finds_an_exact_line_on_the_log_scale():
    points = [(x, 0.1 + 0.2 * math.log(x)) for x in (1, 2, 4, 8)]
    a, b, r = script.lsq_fit(points)
    assert a == pytest.approx(0.1) and b == pytest.approx(0.2) and r == pytest.approx(1.0)
    assert script.fit_value((a, b, r), 1000) == 1.0 # accuracy is clipped to 0..1

def test_lsq_fit_refuses_points_of_a_single_length():
    with pytest.raises(ValueError):
        script.lsq_fit([(2, 0.1), (2, 0.5)])

def test_common_range_is_the_overlap_of_all_groups_or_none():
    assert script.common_range({"a": [(1, 0), (10, 0)], "b": [(2, 0), (20, 0)]}) == (2, 10)
    assert script.common_range({"a": [(1, 0), (2, 0)], "b": [(3, 0), (4, 0)]}) is None

def test_binned_median_skips_bins_with_too_few_points():
    points = [(0.05, 0.1), (0.06, 0.3), (0.07, 0.2), (0.5, 0.9)]
    assert script.binned_median(points, edges=[0.04, 0.08, 1.0], min_n=3) == [(math.sqrt(0.04 * 0.08), 0.2)]

def test_method_and_run_names_come_from_the_compressed_file():
    assert script.method_of("data/compress.SkLLM.toml") == "SkLLM"
    assert script.method_of("data/compress.DSPM.toml") == "DSPM-NoLegend"
    assert script.method_of("data/compress.DSPM.toml", tag_legend=True) == "DSPM+Legend"
    assert script.method_of("data/compress.Frontier.p01.d001.toml") == "Frontier models"
    assert script.method_of("data/compress.Gemma26B.toml") == "Gemma26B direct"
    assert script.run_name("data/compress.Frontier.p01.d001.toml") == "Frontier.p01.d001"

def test_prompts_are_compared_at_one_common_real_length(result_files):
    runs = script.load_runs(result_files)
    dialogues = script.per_dialogue(runs)
    same = script.prompt_same_length(script.prompt_table(runs, dialogues), dialogues)
    s = same[("LLM", "t001")]
    assert s["range"] == pytest.approx((0.2, 0.4)) # p01/p02: 10-40% of the original, p03: 20-50%
    assert s["at"] == pytest.approx(math.sqrt(0.2 * 0.4))
    # p03 writes longer texts, so its plain mean is the highest, but at the same length p02 and p03 are equal
    assert s["acc"]["p02"] == pytest.approx(s["acc"]["p03"], abs=0.02)
    assert s["acc"]["p01"] < s["acc"]["p02"]

def test_report_numbers_come_from_the_files_and_the_notes_go_on_top(result_files):
    runs = script.load_runs(result_files)
    notes = "# Note: hand-written\nKeep this text."
    report = script.build_report(runs, notes=notes)
    assert report.startswith(notes)
    assert "Floor (no text): **1/10 = 10.0%** (8 no answer). Ceiling (original text): **10/10 = 100.0%** (0 no answer)." in report
    assert "## Prompts p01, p02, p03 compared on the same compressor" in report
    assert "![" not in report # no charts were asked for
    assert "| SkLLM | 1 | 8% |" in report

def test_report_shows_reference_points_that_differ_between_runs(result_files, write_toml, read_toml):
    other = read_toml(result_files[1])
    other["results"][0]["correct"] = 2
    runs = script.load_runs([result_files[0], write_toml("benchmark.SkLLM.t001.toml", other)])
    assert script.reference_differences(runs) == [(("t001", "No Text"), [(1, 8), (2, 8)])]
    assert "t001 No Text: [(1, 8), (2, 8)]" in script.build_report(runs)

def test_reply_forms_tell_a_bare_letter_from_other_replies():
    row = lambda reply, parsed, finish="stop": {"reply": reply, "parsed": parsed, "finish": finish}
    assert script.reply_form(row("C", "C")) == "bare letter"
    assert script.reply_form(row("B (Note: the text", "B")) == "letter in a longer reply"
    assert script.reply_form(row("The provided text does not", None)) == "no letter"
    assert script.reply_form(row("", None, "error")) == "failed"

def test_letter_choices_in_a_long_reply():
    assert script.letter_choices("I am not sure. The answer is B.") == ["B"]
    assert script.letter_choices("The text does not say which day it was.") == []

def test_script_writes_the_report_without_charts(result_files, run_main, tmp_path):
    output = tmp_path / "docs" / "report.md"
    notes = tmp_path / "notes.md"
    notes.write_text("# Preliminary\n", encoding="utf-8")
    run_main(script, "--results", str(tmp_path / "benchmark.*.toml"), "--notes", notes, "--output", output, "--no-charts")
    text = output.read_text(encoding="utf-8")
    assert text.startswith("# Preliminary") and "## t001:" in text

def test_script_stops_when_there_are_no_result_files(run_main, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_main(script, "--results", str(tmp_path / "none.*.toml"), "--output", tmp_path / "r.md", "--no-charts")
    assert "No benchmark result files" in capsys.readouterr().err

def test_dspm_with_and_without_the_tag_legend_are_two_systems(write_toml):
    plain  = bench_file("data/compress.DSPM.toml", [result(prompt="NA", tokens=20, correct=3)])
    legend = bench_file("data/compress.DSPM.toml", [result(prompt="NA", tokens=20, correct=5)])
    legend["benchmark"]["tag_legend"] = True
    runs = script.load_runs([write_toml("benchmark.DSPM.t001.toml", plain), write_toml("benchmark.DSPM-Legend.t001.toml", legend)])
    assert sorted((r["run"], r["method"]) for r in runs) == [("DSPM", "DSPM-NoLegend"), ("DSPM+Legend", "DSPM+Legend")]
    report = script.build_report(runs)
    assert "| DSPM-NoLegend | 1 |" in report and "| DSPM+Legend | 1 |" in report
    assert "--tag-legend" in report # the command of the legend run is shown with the option

def test_reference_point_is_the_value_most_runs_got(write_toml, read_toml):
    files = []
    for name, floor in (("benchmark.A.t001.toml", 2), ("benchmark.B.t001.toml", 1), ("benchmark.C.t001.toml", 1)):
        doc = bench_file(f"data/compress.{name.split('.')[1]}.toml", [result(prompt="NA")])
        doc["results"][0]["correct"] = floor
        files.append(write_toml(name, doc))
    dialogues = script.per_dialogue(script.load_runs(files))
    assert dialogues["t001"]["floor"]["correct"] == 1 # two runs got 1, one run (the first file) got 2
