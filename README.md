# DialogKit (WIP)
Generate, manage, and analyze dialogues in Python. Create synthetic multi-party conversations, store them in a clean structured format, and run analytics on turns, speakers, and content. Ideal for NLP research, chatbot testing, and training-data pipelines.

**This project is work in progress (WIP).**

# The Idea
My work in ASR and NLP has repeatedly brought me to dialogues. Before the rise of LLMs, I was fortunate to work extensively with real-world dialogue data, and I've found it a fascinating field, though one that still lacks proven, reliable tooling.

Today, working with dialogues is orders of magnitude easier. Yet I still haven't found a "golden tool", something like a master craftsman for this domain.
So I decided to create a prototype - **DialogKit** is my attempt to build one.

# What Works
It's still early days, and only selected tools are doing anything useful.

## Compress Dialogue (Tool)
`utils/compress_Gemma26B.py` - A utility script that compresses text from data files (e.g. `data/data.d001.toml`) using the Gemma LLM and outputs the results to a structured TOML file.
The script results have been added to the repository at: `data/compress.Gemma26B.toml`.

To use it, first start LLM hosting (as below) or change script settings to use LLM in the cloud.
```
python -m vllm.entrypoints.openai.api_server \
    --model cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
    --quantization compressed-tensors \
    --dtype float16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 16384 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --port 8000
```
Then run the script, which will processes:
- two LLM configurations (without thinking and with thinking),
- three types of compression prompts,
- three sample dialogs,
- 15 different target token counts after compression.

This gives 270 LLM queries (= 2\*3\*3\*15), and this may take some time.

Example script run:
`uv run utils/compress_Gemma26B.py data/prompts.compress.toml "data/data.*.toml" data/compress.Gemma26B.toml`

## Count the Number of Tokens and Words (Tool)
`utils/estimate_tokens.py` - A utility script that enhances a TOML file by automatically calculating and appending word counts and LLM token counts for text entries in data files (e.g. `data/data.d001.toml`) and compressed results (e.g. `data/compress.Gemma26B.toml`).

To use it, first start LLM hosting (as above) or change script settings to use LLM in the cloud.
You can run it on a generated TOML file with compressed prompts. This is usually a very quick.

Example script run:
`uv run python utils/estimate_tokens.py data/compress.Gemma26B.toml data/compress.Gemma26B.new.toml`

# Brain Storm
At this stage, it looks like a wish list. However, it's worth writing down long-term goals and any ideas.
- Generating:
  - Spec-driven generating synthetic dialogues.
  - Controlling personas, roles, intentions, styles, emotions, flow and conversational dynamics.
  - Injecting information, facts and attributes.
  - Simulation of "noise" after ASR (substitutions, deletions, insertions).
  - Multiple language and language switching support.
- Visualization:
  - Conversation flow graph.
  - Dialog viewer (terminal, HTML, MarkDown).
  - Dialog diff.
- Verification & Testing
  - LLM as a judge verifies whether the generated dialogue complies with the specification.
  - Deterministic rules: length, number of turns, number of speakers, participation of speakers in the conversation (dominance), and whether the injected phrases were actually spoken.
- Processing/Analyzing/Detection:
  - PII detection.
  - Annotator/Detector approach.
  - Topic detection.
  - Taxonomy-based processing.
  - Ontology-based processing.
  - Finite State Transducers (FST) processing (optional).
  - JSpeech Grammar Format (JSGF) processing (optional).
 - Transforming:
   - Anonymization.
   - Augmentation: paraphrases, noise, original, change of style or persona.
   - Information injection.
- Integration:
  - JSON/JSONL
  - ASR results as input.
  - Subtitles (SRT/VTT/..).
  - Hugging Face Datasets.
  - spaCy.
  - Ontology (RDF/OWL).
  - Taxonomies (SKOS).
   - Compression (summary, essence).
   - Conversion (knowledge base).
