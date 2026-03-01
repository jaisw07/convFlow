import os
import json
import asyncio
from typing import AsyncGenerator, Dict, Any, List, Optional
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class StreamingInterviewBrain:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        temperature: float = 0.4,
        max_tokens: int = 800,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    async def process_turn(
        self,
        transcript: str,
        state_snapshot: Dict[str, Any],
        short_memory: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Streams Gemini response.
        Accumulates JSON.
        Returns parsed structured output.
        """

        prompt = self._build_prompt(transcript, state_snapshot, short_memory)

        stream = client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        )

        full_response = ""

        async def iterate():
            nonlocal full_response
            for chunk in stream:
                if chunk.text:
                    full_response += chunk.text
                    await asyncio.sleep(0)

        await iterate()

        try:
            parsed = json.loads(full_response)
        except Exception:
            print("⚠️ Failed to parse structured LLM output.")
            return self._fallback_response()

        return parsed

    # ---------------------------------------------------------
    # Prompt Builder
    # ---------------------------------------------------------

    def _build_prompt(
        self,
        transcript: str,
        state: Dict[str, Any],
        memory: List[Dict[str, str]],
    ) -> str:

        history_text = ""
        for turn in memory:
            history_text += f"{turn['role'].capitalize()}: {turn['content']}\n"

        return f"""
You are conducting a professional {state['topic']}.

Current Phase: {state['phase']}
Difficulty Level: {state['difficulty']}

Interview Summary:
{state.get('summary', '')}

Conversation so far:
{history_text}

Candidate's Latest Answer:
{transcript}

Return STRICT JSON only in this format:

{{
  "evaluation": {{
    "technical_depth": 1-10,
    "correctness": 1-10,
    "clarity": 1-10,
    "confidence": 1-10,
    "behavior": "normal | abusive | irrelevant | restart_attempt"
  }},
  "profile_update": {{
    "name": null or string,
    "skills": [],
    "projects": [],
    "experience": []
  }},
  "difficulty_adjustment": "increase | decrease | same",
  "phase_shift": "INTRO | TECHNICAL | BEHAVIORAL | CLOSING | same",
  "next_question": "string"
}}

Rules:
- Do not include markdown.
- Do not explain.
- Only valid JSON.
"""

    # ---------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------

    def _fallback_response(self):
        return {
            "evaluation": {
                "technical_depth": 5,
                "correctness": 5,
                "clarity": 5,
                "confidence": 5,
                "behavior": "normal",
            },
            "profile_update": {
                "name": None,
                "skills": [],
                "projects": [],
                "experience": [],
            },
            "difficulty_adjustment": "same",
            "phase_shift": "same",
            "next_question": "Could you elaborate further on your previous answer?"
        }