import subprocess
import tempfile
import os
import threading
import sounddevice as sd
import soundfile as sf

from audio.tts.base import BaseTTS

SAMPLE_RATE = 16000  # Piper default


class PiperTTS(BaseTTS):
    """
    Piper TTS wrapper (English only).

    Responsibilities:
    - Text → speech audio
    - Play audio
    - Support interruption (stop)
    """

    def __init__(
        self,
        piper_bin: str = "piper",
        voice_model: str = "audio/voices/en_GB-alan-low.onnx.json",
    ):
        self.piper_bin = piper_bin
        self.voice_model = voice_model

        self._play_thread = None
        self._stop_event = threading.Event()

        if not os.path.exists(self.voice_model):
            raise FileNotFoundError(
                f"Piper voice model not found: {self.voice_model}"
            )

        print("[PiperTTS] Initialized.")

    # --------------------------------------------------------------
    # Public API (BaseTTS)
    # --------------------------------------------------------------

    def speak(self, text: str):
        """
        Convert text to speech and play it.
        Non-blocking; playback runs in its own thread.
        """

        if not text.strip():
            return

        # Stop any ongoing playback
        self.stop()

        self._stop_event.clear()

        self._play_thread = threading.Thread(
            target=self._speak_worker,
            args=(text,),
            daemon=True,
        )
        self._play_thread.start()

    def stop(self):
        """
        Interrupt current playback.
        """
        if self._play_thread and self._play_thread.is_alive():
            self._stop_event.set()
            sd.stop()

    # --------------------------------------------------------------
    # Internal
    # --------------------------------------------------------------

    def _speak_worker(self, text: str):
        """
        Run Piper and play audio.
        """

        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as tmp:
            wav_path = tmp.name

        try:
            cmd = [
                self.piper_bin,
                "--model",
                self.voice_model,
                "--output_file",
                wav_path,
            ]

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            process.stdin.write(text.encode("utf-8"))
            process.stdin.close()
            process.wait()

            if self._stop_event.is_set():
                return

            audio, sr = sf.read(wav_path, dtype="float32")

            if sr != SAMPLE_RATE:
                print(
                    f"[PiperTTS] Warning: expected {SAMPLE_RATE} Hz, got {sr}"
                )

            sd.play(audio, sr)
            sd.wait()

        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)