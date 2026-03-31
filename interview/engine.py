import asyncio
from typing import Dict, Any

from interview.init_state import create_initial_state

from interview.nodes.node_b_decide import node_b_decide
from interview.nodes.node_c_generate import node_c_generate_stream
from interview.nodes.rolling_summarizer import rolling_summarize
from interview.utils.summary_utils import summary_trigger


class InterviewEngine:

    def __init__(self, llm):

        self.llm = llm
        self.state = create_initial_state()
    
    async def background_summarize(self):
        summary = self.state["rolling_summary"]
        new_summary = await rolling_summarize(self.llm, summary)
        self.state["rolling_summary"] = new_summary
        print("🧠 Rolling summary compressed.")

    def _is_empty_or_nan(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        return not normalized or normalized in {"nan", "none", "null", "na", "n/a"}

    def _is_repeat_request(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        repeat_patterns = [
            "repeat",
            "say that again",
            "can you repeat",
            "kya bola",
            "phir se",
            "dobara",
        ]
        return any(pattern in normalized for pattern in repeat_patterns)

    def _merge_profile_updates(self, updates: Dict[str, Any]) -> None:
        if not isinstance(updates, dict):
            return
        profile = self.state["user_profile"]
        for key, value in updates.items():
            if key not in profile:
                continue
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if key == "primary_concerns" and isinstance(value, list):
                merged = list(profile.get("primary_concerns", []))
                for concern in value:
                    if concern and concern not in merged:
                        merged.append(concern)
                profile[key] = merged
                continue
            profile[key] = value

    def _sanitize_tts_text(self, text: str) -> str:
        cleaned = (text or "").replace("*", "").replace("#", "")
        cleaned = cleaned.replace("`", "").replace("_", "")
        return " ".join(cleaned.split())

    def _build_context(self, node_b_result: Dict[str, Any], last_q: str, last_a: str) -> str:
        profile = self.state["user_profile"]
        return f"""
Assistant purpose: Type 1 Diabetes awareness and support assistant.

Current phase: {node_b_result.get("phase", "onboarding")}
Intent: {node_b_result.get("intent", "answer_question")}
Topic: {node_b_result.get("topic", "general")}
Reason: {node_b_result.get("reason", "")}
Language style: {node_b_result.get("language_style", "simple")}
Detected input language: {node_b_result.get("input_language", "unknown")}
User type: {profile.get("user_type", "unknown")}

User profile snapshot: {profile}

Last assistant message: {self.state.get("last_assistant_message", "")}
Assistant previous question: {last_q}
Latest user message: {last_a}
"""

    async def stream_step(self, transcript):

        if self.state["phase"] == "onboarding" and not self.state["last_question"]:
            question = (
                "Hi, I am your Type 1 Diabetes support assistant. "
                "Are you a parent or a child with Type 1 Diabetes?"
            )
            self.state["last_question"] = question
            self.state["last_assistant_message"] = question
            yield question
            return

        if self._is_empty_or_nan(transcript):
            clarification = "I did not catch that. Could you please repeat in one short sentence?"
            self.state["last_assistant_message"] = clarification
            self.state["last_question"] = clarification
            yield clarification
            return

        if self._is_repeat_request(transcript) and self.state["last_assistant_message"]:
            repeat_text = self.state["last_assistant_message"]
            self.state["last_question"] = repeat_text
            yield repeat_text
            return

        last_q = self.state["last_question"]
        last_a = transcript

        self.state["last_answer"] = transcript

        # Append new QA
        self.state["rolling_summary"] += f"\nQ:{last_q}\nA:{last_a}\n"

        summary = self.state["rolling_summary"]

        # Trigger background summarization
        if summary_trigger(self.state["rolling_summary"]):
            asyncio.create_task(self.background_summarize())

        node_b_result = await node_b_decide(
            self.llm,
            last_q,
            last_a,
            summary,
            self.state["user_profile"].get("user_type", "unknown"),
            self.state["phase"],
            self.state["user_profile"],
        )

        # Update tracked profile with model inferences.
        proposed_user_type = node_b_result.get("user_type", "unknown")
        confidence = float(node_b_result.get("confidence", 0.0) or 0.0)
        if proposed_user_type in {"parent", "child_patient"} and confidence >= 0.55:
            self.state["user_profile"]["user_type"] = proposed_user_type
            self.state["user_profile"]["profile_confidence"] = confidence
        elif self.state["user_profile"].get("user_type") == "unknown":
            self.state["user_profile"]["profile_confidence"] = max(
                self.state["user_profile"].get("profile_confidence", 0.0),
                confidence,
            )

        input_lang = node_b_result.get("input_language", "unknown")
        if input_lang in {"english", "hindi", "hinglish", "unknown"}:
            self.state["user_profile"]["input_language"] = input_lang

        self._merge_profile_updates(node_b_result.get("profile_updates", {}))

        self.state["phase"] = node_b_result.get("phase", self.state["phase"])

        if not node_b_result.get("understood", True):
            clarification = "I may have missed your words. Please repeat slowly in one short sentence."
            self.state["last_assistant_message"] = clarification
            self.state["last_question"] = clarification
            yield clarification
            return

        if node_b_result.get("intent") == "repeat_last" and self.state["last_assistant_message"]:
            repeat_text = self.state["last_assistant_message"]
            self.state["last_question"] = repeat_text
            yield repeat_text
            return

        context = self._build_context(node_b_result, last_q, last_a)

        question_buffer = ""

        async for token in node_c_generate_stream(self.llm, context):
            clean_token = self._sanitize_tts_text(token)
            question_buffer += clean_token
            if clean_token:
                yield clean_token

        final_response = self._sanitize_tts_text(question_buffer.strip())
        if not final_response:
            final_response = "Could you please repeat your question in a simpler way?"

        self.state["last_question"] = final_response
        self.state["last_assistant_message"] = final_response
        self.state["asked_questions_phase"].append(self.state["last_question"])
