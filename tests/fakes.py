# -*- coding: utf-8 -*-
# About: Stand-ins for the outside world, used by the tests instead of a real LLM server.
#
# Why fakes: the scripts talk to an LLM. A real LLM is slow, needs a GPU and never answers twice the same way,
# so it cannot tell us whether our own code is right. A fake answers at once and exactly as the test tells it to.
# It also remembers every query, so a test can check what the script really sent.
import sys
import types
import importlib.util
from types import SimpleNamespace

class FakeLLM:
    """stand-in for openai.AsyncOpenAI: answers chat queries without any server

    'reply' decides the answer:
      - a string or None:  returned as the answer text for every query,
      - an Exception:      raised, like a failed query,
      - a function:        called with the query arguments (a dict), returns one of the above.
    Every query is kept in 'calls'.
    """
    def __init__(self, reply="OK", finish_reason: str = "stop"):
        self.reply         = reply
        self.finish_reason = finish_reason
        self.calls         = []   # arguments of every chat query, in order
        self.client_kwargs = None # arguments the script used to create the client (base_url, api_key)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def as_client_class(self, **kwargs):
        """use in place of the AsyncOpenAI class: the script 'creates a client' and gets this fake"""
        self.client_kwargs = kwargs
        return self

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.reply(kwargs) if callable(self.reply) else self.reply
        if isinstance(reply, Exception):
            raise reply
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=self.finish_reason)])

class FakeTokenCounter:
    """stand-in for useful_utils.get_precise_token_count: returns a fixed count or one computed from the texts"""
    def __init__(self, count=100):
        self.count = count # an int, or a function (system_text, user_text) -> int
        self.calls = []

    def __call__(self, base_url, model_name, system_content, user_content):
        self.calls.append((base_url, model_name, system_content, user_content))
        return self.count(system_content, user_content) if callable(self.count) else self.count

class FakeResponse:
    """stand-in for requests.Response, as much of it as the scripts use"""
    def __init__(self, json_data=None, error: Exception = None):
        self._json, self._error = json_data, error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        if self._json is None:
            raise ValueError("not a JSON answer")
        return self._json

class _MissingPackage:
    """placeholder for a class from a package that is not installed; tests must replace it before use"""
    def __init__(self, *args, **kwargs):
        raise RuntimeError("This is a placeholder for a package that is not installed. A test must replace it with a fake.")

def install_placeholders_for_missing_packages() -> list:
    """lets the scripts be imported when the LLM packages are not installed, returns names of the replaced packages

    The scripts import their LLM packages at the top (openai, dspm, skllm). 'dspm' and 'skllm' are optional extras,
    installed only for one run (see "How to use it" in each script), so normally they are not there during tests.
    The tests never need the real packages: every test replaces the LLM classes with fakes. So for each package
    that is missing, an empty placeholder is registered, just enough for the 'import' lines to pass.
    A package that is installed is left alone.
    """
    placeholders = {
        "openai": {"openai": ["AsyncOpenAI", "OpenAI"]},
        "dspm":   {"dspm":   ["AsyncDSPMMemory"]},
        "skllm":  {
            "skllm": [], "skllm.config": ["SKLLMConfig"],
            "skllm.models": [], "skllm.models.gpt": [], "skllm.models.gpt.text2text": [],
            "skllm.models.gpt.text2text.summarization": ["GPTSummarizer"],
        },
    }
    replaced = []
    for package, modules in placeholders.items():
        if importlib.util.find_spec(package) is not None:
            continue # the real package is installed
        for module_name, class_names in modules.items():
            module = types.ModuleType(module_name)
            module.__version__ = "placeholder"
            for class_name in class_names:
                setattr(module, class_name, type(class_name, (_MissingPackage,), {}))
            sys.modules[module_name] = module
        replaced.append(package)
    return replaced
