# DialogKit (WIP)
Generate, manage, and analyze dialogues in Python. Create synthetic multi-party conversations, store them in a clean structured format, and run analytics on turns, speakers, and content. Ideal for NLP research, chatbot testing, and training-data pipelines.

**This project is work in progress (WIP).**

# The Idea
My work in ASR and NLP has repeatedly brought me to dialogues. Before the rise of LLMs, I was fortunate to work extensively with real-world dialogue data, and I've found it a fascinating field, though one that still lacks proven, reliable tooling.

Today, generating, analyzing, testing, and managing dialogues is orders of magnitude easier. Yet I still haven't found a "golden tool", something like a master craftsman for this domain.
So I decided to create a prototype - **DialogKit** is my attempt to build one.

# What Works
It's still early days, and only selected tools are doing anything useful.

**compress_Gemma26B.py** - A utility script that compresses text from data files (e.g. data.d001.toml) using the Gemma LLM and outputs the results to a structured TOML file.
The script results have been added to the repository at: `compress.Gemma26B.toml`.
To run it, first start hosting LLM; e.g. with:
```python -m vllm.entrypoints.openai.api_server \
    --model cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
    --quantization compressed-tensors \
    --dtype float16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 16384 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --port 8000
```
Then run the script; e.g.:
`uv run utils/compress_Gemma26B.py data/prompts.compress.toml "data/data.*.toml" data/compress.Gemma26B.toml`

**estimate_tokens.py** - A utility script that enhances a TOML file by automatically calculating and appending word counts and LLM token counts for text entries in data files (e.g. data.d001.toml) and compressed results (e.g. compress.Gemma26B.toml).
You can run it e.g. on a generated TOML file with compressed prompts:
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
