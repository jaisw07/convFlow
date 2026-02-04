import os
from dotenv import load_dotenv
from typing import Optional, List, Dict

import google.generativeai as genai

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment")

genai.configure(api_key=GEMINI_API_KEY)


# ------------------------------------------------------------------
# Gemini LLM Wrapper
# ------------------------------------------------------------------

class GeminiLLM:
    """
    Gemini LLM wrapper for turn-based conversational responses.

    Designed for:
    - Interview agents
    - Low-latency turn response
    - Future TTS integration
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        temperature: float = 0.4,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ):
        self.model_name = model_name

        self.generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        self.system_prompt = system_prompt or self._default_system_prompt()

        print(f"[GeminiLLM] Initializing {model_name}...")

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            system_instruction=self.system_prompt,
        )

        print("[GeminiLLM] Model ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        user_text: str,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate a response for a completed user turn.
        """

        if not user_text.strip():
            return ""

        messages = []

        # --------------------------------------------------
        # Conversation history
        # --------------------------------------------------
        if context:
            for turn in context:
                role = turn.get("role")
                content = turn.get("content", "")

                if not content:
                    continue

                if role == "assistant":
                    gemini_role = "model"
                elif role == "user":
                    gemini_role = "user"
                else:
                    continue

                messages.append(
                    {
                        "role": gemini_role,
                        "parts": [content],
                    }
                )

        # --------------------------------------------------
        # Current user turn
        # --------------------------------------------------
        messages.append(
            {
                "role": "user",
                "parts": [user_text],
            }
        )

        # --------------------------------------------------
        # Gemini call
        # --------------------------------------------------
        response = self.model.generate_content(messages)

        return response.text.strip()

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def _default_system_prompt(self) -> str:
        """
        Interview-optimized system prompt.
        """
        return (
            "You are a professional interview assistant.\n"
            "You ask clear, concise questions and respond naturally.\n"
            "You wait for the candidate to finish speaking before replying.\n"
            "You do not interrupt or speak over the user.\n"
            "Your tone is calm, neutral, and encouraging.\n"
            "Avoid long monologues unless explicitly asked.\n"
            "If an answer is unclear, ask a brief follow-up question.\n"
            "Return your answer in plain text without using any markup tags or formatting or special symbols.\n"
            "Stick to English strictly even if the user responds in Hinglish or Hindi.\n"
        )

# ------------------------------------------------------------------
# LIVE DEMO (Mic → Smart Turn → Whisper → Gemini)
# ------------------------------------------------------------------


""" if __name__ == "__main__":
    import time
    import queue
    import threading
    import numpy as np

    from audio.mic_input import MicInput
    from audio.vad import SileroVAD
    from audio.buffer import TurnBuffer
    from turn_taking.smart_turn import SmartTurnV3
    from stt.whisper_stt import WhisperSTT
    from llm.llm import GeminiLLM

    print("🎤 Speak into the microphone.")
    print("Smart Turn → Whisper → Gemini will respond.")
    print("Press Ctrl+C to stop.\n")

    # --------------------------------------------------------------
    # Initialize components
    # --------------------------------------------------------------

    mic = MicInput()
    vad = SileroVAD()
    buffer = TurnBuffer()
    smart_turn = SmartTurnV3()

    stt = WhisperSTT()
    llm = GeminiLLM()

    llm_queue = queue.Queue()

    conversation_history = []

    # --------------------------------------------------------------
    # Worker: STT → LLM
    # --------------------------------------------------------------

    def llm_worker():
        while True:
            audio = llm_queue.get()
            if audio is None:
                break

            try:
                # ---- STT ----
                transcript = stt.transcribe(audio)

                if not transcript:
                    print("⚠️ Empty transcript, skipping.\n")
                    continue

                print(f"📝 You said:\n{transcript}\n")

                # ---- LLM ----
                response = llm.generate(
                    user_text=transcript,
                    context=conversation_history,
                )

                print(f"🤖 Gemini:\n{response}\n")

                # ---- Save context ----
                conversation_history.append(
                    {"role": "user", "content": transcript}
                )
                conversation_history.append(
                    {"role": "assistant", "content": response}
                )

            finally:
                llm_queue.task_done()

    worker = threading.Thread(target=llm_worker, daemon=True)
    worker.start()

    # --------------------------------------------------------------
    # Audio callback
    # --------------------------------------------------------------

    def on_audio_frame(frame: np.ndarray):
        is_speaking = vad.process_frame(frame)

        if is_speaking:
            buffer.add_speech_frame(frame)
        else:
            buffer.add_silence_frame(frame)

        if buffer.should_check_turn():
            audio_8s = buffer.get_audio_for_smart_turn()
            complete, prob = smart_turn.is_end_of_turn(audio_8s)

            print("🧠 Smart Turn check:")
            print(f"   Complete: {complete}")
            print(f"   Probability: {prob:.4f}")

            if complete:
                raw_audio = buffer.get_full_turn_audio()

                def rms(audio: np.ndarray) -> float:
                    return np.sqrt(np.mean(audio ** 2))

                if rms(raw_audio) < 0.005:
                    print("⚠️ Skipping turn (audio too quiet)\n")
                    buffer.reset()
                    return

                print("✅ Turn accepted. Sending to STT + LLM...\n")

                llm_queue.put(raw_audio)
                buffer.reset()
            else:
                print("⏳ Not complete yet. Listening...\n")

    # --------------------------------------------------------------
    # Start mic
    # --------------------------------------------------------------

    mic.start(on_audio_frame)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic.stop()
        llm_queue.put(None)
        llm_queue.join()
 """