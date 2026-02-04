import os
from dotenv import load_dotenv
from typing import Optional, List, Dict

from google import genai

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment")

client = genai.Client(api_key=GEMINI_API_KEY)

# ------------------------------------------------------------------
# Gemini LLM Wrapper
# ------------------------------------------------------------------

class GeminiLLM:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        temperature: float = 0.4,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or self._default_system_prompt()

        print(f"[GeminiLLM] Initializing {model_name}...")
        print("[GeminiLLM] Model ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        user_text: str,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> str:

        if not user_text.strip():
            return ""

        # --------------------------------------------------
        # Build conversation context as plain text
        # --------------------------------------------------

        history_text = ""

        if context:
            for turn in context:
                role = turn.get("role")
                content = turn.get("content", "").strip()
                if not content:
                    continue

                if role == "user":
                    history_text += f"User: {content}\n"
                elif role == "assistant":
                    history_text += f"Assistant: {content}\n"

        prompt = (
            f"{history_text}"
            f"User: {user_text}\n"
            f"Assistant:"
        )

        # --------------------------------------------------
        # Gemini call (CORRECT SHAPE)
        # --------------------------------------------------

        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,   # ✅ STRING, not list
            config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "system_instruction": self.system_prompt,
            },
        )

        return response.text.strip()


    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def _default_system_prompt(self) -> str:
        return (
            "You are a professional interview assistant.\n"
            "You ask clear, concise questions and respond naturally.\n"
            "You wait for the candidate to finish speaking before replying.\n"
            "You do not interrupt or speak over the user.\n"
            "Your tone is calm, neutral, and encouraging.\n"
            "Avoid long monologues unless explicitly asked.\n"
            "If an answer is unclear, ask a brief follow-up question.\n"
            "Return your answer in plain text without using any markup.\n"
            "Stick to English strictly even if the user responds in Hinglish or Hindi.\n"
        )