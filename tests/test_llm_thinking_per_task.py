"""Which LLM tasks run with the model's reasoning ("thinking") switched off."""
from types import SimpleNamespace

from topic_overviews.kg.topics import Topic
from topic_overviews.llm.keyworder import keywords_paper
from topic_overviews.llm.summarizer import summarize_paper
from topic_overviews.llm.topic_classifier import classify_paper


class RecordingLLM:
    def __init__(self, answer):
        self.answer = answer
        self.thinking = []

    def complete(self, prompt, *, model, max_tokens, thinking=True):
        self.thinking.append(thinking)
        return self.answer


PAPER = SimpleNamespace(title="T", abstract="A")


def test_tldr_runs_without_thinking():
    llm = RecordingLLM("Two sentences.")
    summarize_paper(PAPER, model="m", api_key="", llm=llm)
    assert llm.thinking == [False]


def test_keywords_run_without_thinking():
    llm = RecordingLLM('["A", "B"]')
    keywords_paper(PAPER, model="m", api_key="", llm=llm)
    assert llm.thinking == [False]


def test_classification_keeps_thinking():
    # Thinking off misclassified borderline papers in a comparison test
    # (e.g. multi-agent RL put into "ABM and opinion dynamics").
    llm = RecordingLLM('{"topics": ["Q1"]}')
    classify_paper(PAPER, [Topic("Q1", "T", "D")], model="m", api_key="", llm=llm)
    assert llm.thinking == [True]
