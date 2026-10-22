"""LLM factory and prompt contracts."""

from __future__ import annotations

import pytest

from agent import llm
from agent import prompts as p


@pytest.fixture
def clean_llm(monkeypatch):
    for name in ("LLM_PROVIDER", "MODEL_NAME", "LLM_TEMPERATURE"):
        monkeypatch.delenv(name, raising=False)
    llm.reset_chat_model_cache()
    created = []
    monkeypatch.setattr(llm, "init_chat_model",
                        lambda name, **kw: created.append((name, kw)) or object())
    yield created
    llm.reset_chat_model_cache()


class TestGetChatModel:
    def test_defaults_to_google_genai(self, clean_llm):
        llm.get_chat_model()
        assert clean_llm == [("gemini-2.0-flash", {"model_provider": "google_genai"})]

    @pytest.mark.parametrize("provider", ["openai", "anthropic", "google_genai", "xai"])
    def test_each_provider_has_a_default_model(self, clean_llm, monkeypatch, provider):
        monkeypatch.setenv("LLM_PROVIDER", provider)
        llm.get_chat_model()
        name, kwargs = clean_llm[-1]
        assert name == llm.DEFAULT_MODELS[provider] and kwargs["model_provider"] == provider

    def test_model_name_overrides_default(self, clean_llm, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("MODEL_NAME", "my-model")
        llm.get_chat_model()
        assert clean_llm[-1][0] == "my-model"

    @pytest.mark.parametrize("alias,expected", [("Gemini", "google_genai"), ("grok", "xai"),
                                                ("CLAUDE", "anthropic"), (" openai ", "openai")])
    def test_aliases_and_normalisation(self, clean_llm, monkeypatch, alias, expected):
        monkeypatch.setenv("LLM_PROVIDER", alias)
        assert llm.get_provider() == expected

    def test_model_is_cached_not_recreated(self, clean_llm):
        first = llm.get_chat_model()
        assert llm.get_chat_model() is first and len(clean_llm) == 1

    def test_changed_configuration_creates_a_new_model(self, clean_llm, monkeypatch):
        first = llm.get_chat_model()
        monkeypatch.setenv("MODEL_NAME", "other")
        assert llm.get_chat_model() is not first and len(clean_llm) == 2

    def test_unknown_provider_needs_model_name(self, clean_llm, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "mistralai")
        with pytest.raises(ValueError, match="MODEL_NAME"):
            llm.get_chat_model()
        monkeypatch.setenv("MODEL_NAME", "mistral-large")
        llm.get_chat_model()
        assert clean_llm[-1] == ("mistral-large", {"model_provider": "mistralai"})

    def test_temperature_is_optional(self, clean_llm, monkeypatch):
        llm.get_chat_model()
        assert "temperature" not in clean_llm[-1][1]
        monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
        llm.get_chat_model()
        assert clean_llm[-1][1]["temperature"] == 0.2

    def test_describe_llm(self, clean_llm, monkeypatch):
        assert llm.describe_llm() == "google_genai:gemini-2.0-flash"
        monkeypatch.setenv("LLM_PROVIDER", "nope")
        assert llm.describe_llm().startswith("unconfigured")
        assert clean_llm == []  # describing never initialises a model


def test_content_to_text():
    assert llm.content_to_text("plain") == "plain"
    assert llm.content_to_text(None) == ""
    blocks = [{"type": "text", "text": "a"}, {"type": "tool_use", "id": "x"}, "b", {"text": "c"}]
    assert llm.content_to_text(blocks) == "abc"


class TestPrompts:
    def test_system_prompt_covers_required_behaviour(self):
        text = p.SYSTEM_PROMPT
        assert "{workspace}" in text and "autonomous coding agent" in text
        for needle in ("read_agent_state", "update_plan", "update_todo", "Inspect files",
                       "incrementally", "run_tests", "install_packages", "review_changes",
                       "Git", "save_memory", "outside the workspace", "pretend",
                       "standard library", "rejected"):
            assert needle.lower() in text.lower(), needle

    def test_render_prompt_only_replaces_workspace(self):
        out = p.render_prompt("root {workspace} and {other}", "/w")
        assert out == "root /w and {other}"

    @pytest.mark.parametrize("stage", ["planner", "coder", "reviewer", "committer"])
    def test_stage_prompts(self, stage):
        text = p.STAGE_PROMPTS[stage]
        assert f"ROLE: {stage.upper()}" in text.splitlines()[0] and "{workspace}" in text
        assert "outside the workspace" in text and "rejected" in text.lower()
        assert stage in p.HANDOFF

    def test_review_prompt_demands_a_verdict_line(self):
        assert "VERDICT: APPROVE" in p.REVIEW_PROMPT and "VERDICT: REQUEST_CHANGES" in p.REVIEW_PROMPT

    def test_stage_prompts_mention_only_their_own_powers(self):
        assert "cannot review or commit" in p.CODER_PROMPT
        assert "read-only" in p.REVIEWER_PROMPT
        assert "write code" in p.PLANNER_PROMPT
