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

    tts = create_tts(
        engine="piper",
        piper_bin=r"C:\Users\SHREY\Desktop\SpeechToText\piper_windows_amd64\piper\piper.exe",
        voice_model="audio/voices/en_GB-alan-low.onnx",
    )

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

                # ------------------ LLM ------------------
                response = llm.generate(
                    user_text=transcript,
                    context=conversation_history,
                )

                if not response:
                    print("⚠️ Empty LLM response.\n")
                    continue

                print(f"🤖 Gemini:\n{response}\n")

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

            if complete:
                raw_audio = buffer.get_full_turn_audio()

                def rms(audio: np.ndarray) -> float:
                    return np.sqrt(np.mean(audio ** 2))

                if rms(raw_audio) < RMS_SILENCE_THRESHOLD:
                    print("⚠️ Skipping turn (audio too quiet)\n")
                    buffer.reset()
                    return

                print("✅ Turn accepted. Processing...\n")

                turn_queue.put(raw_audio)
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
        turn_queue.put(None)
        turn_queue.join()