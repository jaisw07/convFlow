"""
Orchestrator: Mic → VAD → Smart Turn → Whisper → Gemini → Piper TTS

This is the main entrypoint that ties the full pipeline together.
Barge-in is intentionally NOT implemented yet.
"""

import time
import queue
import threading
import numpy as np
import random

from audio.mic_input import MicInput
from audio.vad import SileroVAD
from audio.buffer import TurnBuffer
from turn_taking.smart_turn import SmartTurnV3
from stt.whisper_stt import WhisperSTT
from llm.llm import GeminiLLM
from audio.tts.factory import create_tts

from state.state import InterviewState
from state.evaluator import AnswerEvaluator


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

RMS_SILENCE_THRESHOLD = 0.005


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    # --------------------------------------------------------------
    # Initialize components
    # --------------------------------------------------------------

    mic = MicInput()
    vad = SileroVAD()
    buffer = TurnBuffer()
    smart_turn = SmartTurnV3()

    stt = WhisperSTT()
    llm = GeminiLLM()
    evaluator = AnswerEvaluator()
    state = InterviewState(topic="SDE Intern Interview")

    # tts = create_tts(
    #     engine="piper",
    #     piper_bin=r"C:\Users\SHREY\Desktop\SpeechToText\piper_windows_amd64\piper\piper.exe",
    #     voice_model="audio/voices/en_GB-alan-low.onnx",
    # )

    tts = create_tts(
        engine="kokoro",
        lang_code="a",
        voice="af_heart",
        speed=1.0,
    )

    # tts = create_tts(
    #     engine="f5",
    #     ref_audio_path=r"C:\Users\SHREY\Documents\Sound Recordings\tanmay_sample.wav",
    #     ref_text="I'm just going to tell you my life's journey in very simple words, and which may not leave you inspired, but will help you survive this life.",
    #     device="cuda",
    #     speed = 1.0,
    #     nfe_step=24,   # lower = faster, slightly lower quality
    # )


    tts_busy = threading.Event()

    def on_tts_done():
        print("TTS Finished, Resuming Mic\n")
        tts_busy.clear()

    tts.on_done = on_tts_done

    # Queue of completed turns (audio)
    turn_queue = queue.Queue()

    # Conversation memory
    conversation_history = []
    
    # check valid transcript or not
    def is_valid_transcript(text: str) -> bool:
        if not text:
            return False

        stripped = text.strip()

        if not stripped:
            return False

        lowered = stripped.lower()

        # Reject common junk outputs
        if lowered in {"nan", "none", "null"}:
            return False

        # Reject very short noise
        if len(stripped) < 2:
            return False

        return True

    def decide_mode(state: InterviewState) -> str:
        evaluation = state.last_evaluation

        if not evaluation:
            return "normal"

        correctness = evaluation.get("correctness", 5)
        depth = evaluation.get("technical_depth", 5)

        if correctness < 4:
            return "clarify_mistake"
        
        if state.followup_stack:
            return "use_followup"

        if depth < 5:
            return "probe_deeper"

        return "continue_topic"


    # --------------------------------------------------------------
    # Worker: STT → LLM → TTS
    # --------------------------------------------------------------

    def turn_worker():
        while True:
            audio = turn_queue.get()
            if audio is None:
                break

            try:
                # ------------------ STT ------------------
                transcript = stt.transcribe(audio)

                if not is_valid_transcript(transcript):
                    print("⚠️ Invalid transcript detected.\n")
                    clarifications = [
                        "I'm sorry, I didn't quite catch that. Could you please repeat your response?",
                        "Apologies, there seems to have been some audio disruption. Would you mind repeating that?",
                        "I may have missed your last statement. Could you clarify your answer?",
                    ]
                    choice = random.choice(clarifications)
                    print(f"🤖 Gemini (clarification):\n{choice}\n")
                    tts_busy.set()
                    tts.speak(choice)
                    continue

                print(f"📝 You said:\n{transcript}\n")
                # ------------------ EVALUATION ------------------

                if state.current_question is None:
                    # first turn: no evaluation
                    eval_result = {
                        "technical_depth": 5,
                        "correctness": 5,
                        "clarity": 5,
                        "confidence": 5,
                        "red_flags": [],
                        "followup_opportunities": []
                    }
                else:
                    eval_result = evaluator.evaluate(
                        question=state.current_question,
                        answer=transcript,
                    )

                print(f"📊 Evaluation: {eval_result}\n")

                # 1️⃣ Record turn FIRST
                state.record_turn(
                    question=state.current_question or "Opening Question",
                    answer=transcript,
                    evaluation=eval_result,
                )

                # 2️⃣ Adjust difficulty
                state.adjust_difficulty()

                # 3️⃣ Now decide mode (uses updated evaluation)
                mode = decide_mode(state)

                followup_instruction = ""
                if mode == "use_followup" and state.followup_stack:
                    followup = state.followup_stack.pop(-1)
                    followup_instruction = f"Ask this follow-up question: {followup}"

                # 4️⃣ Now build prompt
                adaptive_prompt = f"""
                    You are conducting a {state.topic}.

                    Current difficulty level (1 is easiest, linear scale): {state.difficulty}
                    Branching mode: {mode}

                    Rules:
                    - Do NOT repeat previous questions.
                    - Be concise.
                    - Stay natural and conversational.

                    Branching behavior:
                    - clarify_mistake → Ask the candidate to correct or reconsider their answer.
                    - probe_deeper → Ask a deeper technical follow-up.
                    - increase_difficulty → Ask a harder, more advanced question.
                    - continue_topic → Continue the same topic naturally.
                    - use_followup → Use the provided follow-up question.

                    {followup_instruction}

                    Ask the next question.
                """

                response = llm.generate(
                    user_text=adaptive_prompt,
                    context=conversation_history,
                )

                if not response:
                    print("⚠️ Empty LLM response.\n")
                    continue

                print(f"🤖 Gemini:\n{response}\n")


                state.current_question = response


                # ------------------ TTS ------------------
                print("Pausing mic. TTS speaking.\n")
                tts_busy.set()
                tts.speak(response)

                # ------------------ Memory ------------------
                conversation_history.append(
                    {"role": "user", "content": transcript}
                )
                conversation_history.append(
                    {"role": "assistant", "content": response}
                )

            finally:
                turn_queue.task_done()

    worker = threading.Thread(target=turn_worker, daemon=True)
    worker.start()

    # --------------------------------------------------------------
    # Audio callback
    # --------------------------------------------------------------
    from collections import deque

    smart_turn_probs = deque(maxlen=5)
    SMART_TURN_THRESHOLD = 0.75
    REQUIRED_HISTORY = 5

    pending_turn = {
        "active": False,
        "timestamp": 0.0,
        "audio": None
    }

    CONFIRM_WINDOW = 0.5 # 500ms

    FAILSAFE_SILENCE_SECONDS = 3
    FRAME_DURATION_SEC = 0.032
    FAILSAFE_SILENCE_FRAMES = int(FAILSAFE_SILENCE_SECONDS / FRAME_DURATION_SEC)


    def on_audio_frame(frame: np.ndarray):
        # ignore mic while tts speaking
        if tts_busy.is_set():
            return
        
        is_speaking = vad.process_frame(frame)

        if is_speaking:
            buffer.add_speech_frame(frame)
        else:
            buffer.add_silence_frame(frame)

        # -------------------------------------------------
        # Hard silence failsafe (4s silence → force commit)
        # -------------------------------------------------

        if (
            buffer.speech_samples >= buffer.min_speech_samples
            and buffer._silent_frames >= FAILSAFE_SILENCE_FRAMES
            and not pending_turn["active"]
        ):
            print("⏱ Failsafe triggered (4s silence). Forcing commit.")

            raw_audio = buffer.get_full_turn_audio()

            def rms(audio: np.ndarray) -> float:
                return np.sqrt(np.mean(audio ** 2))

            if rms(raw_audio) >= RMS_SILENCE_THRESHOLD:
                print(f"Turn duration: {len(raw_audio)/16000:.2f} sec")
                turn_queue.put(raw_audio)

            buffer.reset()
            smart_turn_probs.clear()
            pending_turn["active"] = False
            return


        if buffer.should_check_turn():
            audio_8s = buffer.get_audio_for_smart_turn()
            _, prob = smart_turn.is_end_of_turn(audio_8s)

            smart_turn_probs.append(prob)
            avg_prob = sum(smart_turn_probs) / len(smart_turn_probs)

            print("🧠 Smart Turn check:")
            print(f"   Instant Prob: {prob:.4f}")
            print(f"   Avg Prob: {avg_prob:.4f}")


            if (
                len(smart_turn_probs) == REQUIRED_HISTORY
                and avg_prob >= SMART_TURN_THRESHOLD
                and not pending_turn["active"]
            ):
                print("🟡 SmartTurn suggests turn end. Awaiting confirmation...")
                pending_turn["active"] = True
                pending_turn["timestamp"] = time.time()
                pending_turn["audio"] = buffer.get_full_turn_audio()
                return
            
            # ----------------------------------------
            # Confirm pending turn if silence continues
            # ----------------------------------------

            if pending_turn["active"]:
                elapsed = time.time() - pending_turn["timestamp"]

                if elapsed >= CONFIRM_WINDOW:
                    # If still silent → commit
                    if (
                        buffer._silent_frames >= buffer.silence_trigger_frames
                        and avg_prob >= SMART_TURN_THRESHOLD
                        ):
                        raw_audio = pending_turn["audio"]

                        def rms(audio: np.ndarray) -> float:
                            return np.sqrt(np.mean(audio ** 2))

                        if rms(raw_audio) < RMS_SILENCE_THRESHOLD:
                            print("⚠️ Skipping turn (audio too quiet)\n")
                            buffer.reset()
                        else:
                            print("✅ Turn confirmed. Processing...\n")
                            print(f"Turn duration: {len(raw_audio)/16000:.2f} sec")
                            turn_queue.put(raw_audio) 
                            buffer.reset()

                    else:
                        print("⚠️ Speech resumed. Cancelling turn.\n")
                        smart_turn_probs.clear()

                    pending_turn["active"] = False
                    smart_turn_probs.clear()



    # --------------------------------------------------------------
    # Start mic
    # --------------------------------------------------------------

    mic.start(on_audio_frame)

    
    # --- Add your welcome TTS here ---
    welcome_message = f"Welcome to your {state.topic}. Let's start off with a brief introduction about yourself."
    tts_busy.set()
    tts.speak(welcome_message)
    state.current_question = welcome_message

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic.stop()
        turn_queue.put(None)
        turn_queue.join()