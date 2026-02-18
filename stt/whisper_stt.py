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
        # self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.dtype = torch.float32

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
        if audio.dtype != np.float32:
            raise ValueError("WhisperSTT expects float32 audio")

        if audio.ndim != 1:
            raise ValueError("WhisperSTT expects mono (1D) audio")

        total_seconds = len(audio) / SAMPLE_RATE
        print(f"Input audio seconds: {total_seconds:.2f}")

        # --- If short enough, do single pass ---
        if total_seconds <= 28:
            return self._transcribe_chunk(audio)

        # --- Chunking mode ---
        chunk_size_sec = 25
        overlap_sec = 2

        chunk_size = chunk_size_sec * SAMPLE_RATE
        overlap = overlap_sec * SAMPLE_RATE

        transcripts = []
        start = 0

        while start < len(audio):
            end = min(start + chunk_size, len(audio))
            chunk = audio[start:end]

            print(f"Transcribing chunk {start/SAMPLE_RATE:.2f}s → {end/SAMPLE_RATE:.2f}s")

            text = self._transcribe_chunk(chunk)
            transcripts.append(text)

            if end == len(audio):
                break

            start = end - overlap

        return self._stitch_transcripts(transcripts)
    
    def _transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        inputs = self.processor(
            audio_chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(self.device, dtype=self.dtype)

        generated_ids = self.model.generate(
            input_features,
            task="transcribe",
            language="en",
        )

        text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return text.strip()

    def _stitch_transcripts(self, chunks):
        if not chunks:
            return ""

        final_text = chunks[0]

        for next_chunk in chunks[1:]:
            final_text = self._merge_overlap(final_text, next_chunk)

        return final_text.strip()

    def _merge_overlap(self, prev_text, next_text):
        prev_words = prev_text.split()
        next_words = next_text.split()

        max_overlap = min(len(prev_words), len(next_words), 30)

        for i in range(max_overlap, 0, -1):
            if prev_words[-i:] == next_words[:i]:
                return " ".join(prev_words + next_words[i:])

        return prev_text + " " + next_text


# ------------------------------------------------------------------
# LIVE DEMO (turn-taking + STT)
# ------------------------------------------------------------------

""" if __name__ == "__main__":
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
                raw_audio = buffer.get_full_turn_audio()

                def rms(audio: np.ndarray) -> float:
                    return np.sqrt(np.mean(audio ** 2))

                if rms(raw_audio) < 0.005:
                    print("⚠️ Skipping STT (audio too quiet)")
                    buffer.reset()
                    return

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
        stt_queue.join() """