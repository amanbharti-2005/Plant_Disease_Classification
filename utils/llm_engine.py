"""
LLM-driven diagnosis + Q&A for AgroVision — now running on Groq instead
of Gemini.

Setup:
    pip install groq
    setx GROQ_API_KEY "your-key-here"   (Windows, new terminal after)
    # or: $env:GROQ_API_KEY="your-key-here"  (current PowerShell session)

Get a free key at: https://console.groq.com/keys

IMPORTANT DIFFERENCE FROM THE GEMINI VERSION:
Groq does NOT have a built-in live Google Search grounding tool like
Gemini did. That means answer_question() below can no longer ground
its answers in current web results (prices, regional outbreaks, etc.)
— it now answers purely from the model's own training knowledge, same
as any other LLM call. If live-search-grounded answers matter for your
project, you'd need to add a separate search API (e.g. Tavily, SerpAPI)
and manually feed the results into the prompt. This file does NOT do
that yet — it's flagged here so it doesn't surprise you later.
"""

import os
from dotenv import load_dotenv
load_dotenv(override=True)
import json
import re
from groq import Groq

_client = None


def _get_client():
    """Lazy singleton — avoids crashing the whole API at import time if
    the key isn't set yet; the error only surfaces when actually called."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com/keys"
            )
        _client = Groq(api_key=api_key)
    return _client


# Groq-hosted model. llama-3.3-70b-versatile was deprecated/removed from
# Groq's platform, so this now points at their recommended replacement.
# Swap to "openai/gpt-oss-20b" instead if you need lower latency and can
# accept slightly less detailed answers. Groq's model lineup changes
# fairly often — if this ever 404s again, check the live list at
# https://api.groq.com/openai/v1/models (with your API key) before
# assuming it's a code bug.
DIAGNOSIS_MODEL = "openai/gpt-oss-120b"

FALLBACK_DIAGNOSIS = {
    "severity": "Unknown",
    "recommendations": ["Could not reach the diagnosis service — consult a local agricultural expert."],
    "treatment": "Could not reach the diagnosis service — consult a local agricultural expert.",
}


def _extract_json(text: str) -> dict:
    """Models sometimes wrap JSON in ```json fences even when asked not to."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def get_diagnosis(species: str, condition: str, confidence: float) -> dict:
    """
    Calls the LLM with the species + condition the vision pipeline detected
    and asks it to produce severity/recommendations/treatment from its own
    agronomy knowledge — no lookup table involved, so any crop/disease the
    classifier is ever retrained on works automatically.
    """
    if condition.strip().lower() in ("healthy", "no target object spotted", "invalid image"):
        return {
            "severity": "CLEAR",
            "recommendations": ["Plant appears healthy.", "Continue routine monitoring."],
            "treatment": "No treatment required. Continue standard care and watering schedule.",
        }

    prompt = f"""You are an agronomy expert advising a farmer using an automated
crop-scanning rover. The rover's vision pipeline detected:

Species: {species}
Condition: {condition}
Model confidence: {confidence * 100:.1f}%

Respond with ONLY a raw JSON object (no markdown fences, no commentary) in
exactly this shape:
{{
  "severity": "Low" | "Moderate" | "High",
  "recommendations": ["short actionable bullet", "short actionable bullet", "short actionable bullet"],
  "treatment": "2-4 sentence practical treatment plan a farmer can act on today, naming actual products/methods where appropriate"
}}"""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=DIAGNOSIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content
        data = _extract_json(raw_text)
        return {
            "severity": data.get("severity", "Unknown"),
            "recommendations": data.get("recommendations", []),
            "treatment": data.get("treatment", "No treatment plan returned."),
        }
    except Exception as e:
        print(f"[llm_engine] get_diagnosis failed: {e}")
        return FALLBACK_DIAGNOSIS


def answer_question(species: str, condition: str, question: str) -> str:
    """
    Open-ended Q&A for follow-up farmer questions.

    NOTE: unlike the old Gemini version, this is NOT grounded in live web
    search — Groq has no built-in search tool. Answers come purely from
    the model's training knowledge, so treat anything time-sensitive
    (current prices, this season's outbreaks, product availability) as
    potentially outdated. See the module docstring for how to add real
    search grounding later if you need it.
    """
    prompt = f"""A farmer is looking at a scan result from a crop-monitoring
rover: species = {species}, condition = {condition}.

Their question: {question}

Answer clearly and practically in under 150 words. If the question needs
current information (prices, regional outbreaks, product availability,
recent research) that you cannot verify, say so plainly instead of
guessing, and suggest the farmer check a local source."""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=DIAGNOSIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[llm_engine] answer_question failed: {e}")
        return "Sorry, I couldn't reach the AI service to answer that right now. Please check the GROQ_API_KEY setup and try again."