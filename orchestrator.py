"""
Orchestrator: Mic → VAD → Smart Turn → Whisper → Gemini → Piper TTS

This is the main entrypoint that ties the full pipeline together.
Barge-in is intentionally NOT implemented yet.
"""

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

    print("🎤 Speak into the microphone.")
    print("Mic → Smart Turn → Whisper → Gemini → Piper TTS")
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
    evaluator = AnswerEvaluator()
    state = InterviewState(topic="Technical Interview")

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

                if not transcript:
                    print("⚠️ Empty transcript, skipping.\n")
                    continue

                print(f"📝 You said:\n{transcript}\n")
                # ------------------ EVALUATION ------------------

                if state.current_question is None:
                    # first turn: no evaluation
                    eval_result = {
                        "score": 0.5,
                        "depth": "medium",
                        "feedback": "First turn"
                    }
                else:
                    eval_result = evaluator.evaluate(
                        question=state.current_question,
                        answer=transcript,
                    )

                print(f"📊 Evaluation: {eval_result}\n")

                state.adjust_difficulty()

                # ------------------ LLM (Adaptive Prompt)  ------------------
                adaptive_prompt = f"""
                You are conducting a {state.topic}.

                Current difficulty level: {state.difficulty}
                Average score so far: {state.average_score():.2f}

                Last evaluation:
                Score: {eval_result.get("score")}
                Depth: {eval_result.get("depth")}
                Feedback: {eval_result.get("feedback")}

                Based on the candidate's performance, ask the next best interview question.
                Keep it concise.
                """

                response = llm.generate(
                    user_text=adaptive_prompt,
                    context=conversation_history,
                )

                if not response:
                    print("⚠️ Empty LLM response.\n")
                    continue

                print(f"🤖 Gemini:\n{response}\n")

                state.record_turn(
                    question=state.current_question or "Opening Question",
                    answer=transcript,
                    evaluation=eval_result,
                )

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
    pending_turn = {
        "active": False,
        "timestamp": 0.0,
        "audio": None
    }

    CONFIRM_WINDOW = 0.25  # 250ms


    def on_audio_frame(frame: np.ndarray):
        # ignore mic while tts speaking
        if tts_busy.is_set():
            return
        
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

            if complete and not pending_turn["active"]:
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
                    if buffer._silent_frames >= buffer.silence_trigger_frames:
                        raw_audio = pending_turn["audio"]

                        def rms(audio: np.ndarray) -> float:
                            return np.sqrt(np.mean(audio ** 2))

                        if rms(raw_audio) < RMS_SILENCE_THRESHOLD:
                            print("⚠️ Skipping turn (audio too quiet)\n")
                            buffer.reset()
                        else:
                            print("✅ Turn confirmed. Processing...\n")
                            turn_queue.put(raw_audio)
                            buffer.reset()

                    else:
                        print("⚠️ Speech resumed. Cancelling turn.\n")

                    pending_turn["active"] = False



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
        turn_queue.put(None)
        turn_queue.join()