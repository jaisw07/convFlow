# stt/whisper_stt.py

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

SAMPLE_RATE = 16000


class WhisperSTT:
    """
    Whisper STT wrapper for turn-based transcription.
    Uses Oriserve Whisper Hindi → Hinglish model.
    """

    def __init__(
        self,
        model_id: str = "Oriserve/Whisper-Hindi2Hinglish-Apex",
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"[WhisperSTT] Loading model on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)

        self.model.eval()

        print("[WhisperSTT] Model loaded.")

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a single completed user turn.

        Args:
            audio:
                - np.ndarray
                - float32
                - mono
                - 16kHz
        """

        if audio.dtype != np.float32:
            raise ValueError("WhisperSTT expects float32 audio")

        if audio.ndim != 1:
            raise ValueError("WhisperSTT expects mono (1D) audio")

        # Feature extraction
        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(self.device, dtype=self.dtype)

        # Generate tokens
        generated_ids = self.model.generate(
            input_features,
            task="transcribe",
            language="en",     # Hinglish works best with "en"
        )

        # Decode tokens → text
        text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return text.strip()


# ------------------------------------------------------------------
# LIVE DEMO (turn-taking + STT)
# ------------------------------------------------------------------

if __name__ == "__main__":
    import time
    import queue
    import threading

    from audio.mic_input import MicInput
    from audio.vad import SileroVAD
    from audio.buffer import TurnBuffer
    from turn_taking.smart_turn import SmartTurnV3

    print("🎤 Speak into the microphone.")
    print("Smart Turn + Whisper will transcribe full turns.")
    print("Press Ctrl+C to stop.\n")

    # --- Initialize components ---
    mic = MicInput()
    vad = SileroVAD()
    buffer = TurnBuffer()
    smart_turn = SmartTurnV3()
    stt = WhisperSTT()

    stt_queue = queue.Queue()

    def stt_worker():
        while True:
            audio = stt_queue.get()
            if audio is None:
                break

            try:
                text = stt.transcribe(audio)
                if text:
                    print(f"📝 Transcript:\n{text}\n")
            finally:
                stt_queue.task_done()

    worker = threading.Thread(target=stt_worker, daemon=True)
    worker.start()

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
                print("✅ Turn accepted. Transcribing...\n")

                # Strip Smart Turn padding
                nonzero = np.nonzero(audio_8s)[0]
                raw_audio = audio_8s[nonzero[0]:] if len(nonzero) > 0 else audio_8s

                stt_queue.put(raw_audio)
                buffer.reset()
            else:
                print("⏳ Not complete yet. Continuing to listen...\n")

    mic.start(on_audio_frame)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic.stop()
        stt_queue.put(None)
        stt_queue.join()