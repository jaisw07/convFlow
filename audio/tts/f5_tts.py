import subprocess
import tempfile
import os
import threading
import sounddevice as sd
import soundfile as sf

from audio.tts.base import BaseTTS


class F5TTS(BaseTTS):
    """
    F5-TTS CLI wrapper for voice cloning.

    Note:
    - Reference audio is reused each call.
    - CLI re-encodes reference each time.
    """

    def __init__(
        self,
        ref_audio_path: str,
        ref_text: str,
        model_name: str = "F5TTS_v1_Base",
        device: str = "cuda",
        nfe_step: int = 32,
        speed: float = 1.0,
    ):
        super().__init__()

        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(
                f"Reference audio not found: {ref_audio_path}"
            )

        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.model_name = model_name
        self.device = device
        self.nfe_step = nfe_step
        self.speed = speed

        self._play_thread = None
        self._stop_event = threading.Event()

        print("[F5TTS] Initialized (CLI mode).")

    # --------------------------------------------------------------

    def speak(self, text: str):
        if not text.strip():
            return

        self.stop()
        self._stop_event.clear()

        self._play_thread = threading.Thread(
            target=self._speak_worker,
            args=(text,),
            daemon=True,
        )
        self._play_thread.start()

    def stop(self):
        if self._play_thread and self._play_thread.is_alive():
            self._stop_event.set()
            sd.stop()

    # --------------------------------------------------------------

    def _speak_worker(self, gen_text: str):

        with tempfile.TemporaryDirectory() as tmpdir:

            output_file = "output.wav"

            cmd = [
                "f5-tts_infer-cli",
                "--model", self.model_name,
                "--ref_audio", self.ref_audio_path,
                "--ref_text", self.ref_text,
                "--gen_text", gen_text,
                "--output_dir", tmpdir,
                "--output_file", output_file,
                "--device", self.device,
                "--nfe_step", str(self.nfe_step),
                "--speed", str(self.speed),
            ]

            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                process.wait()

                if self._stop_event.is_set():
                    return

                wav_path = os.path.join(tmpdir, output_file)

                if not os.path.exists(wav_path):
                    print("[F5TTS] Output file not found.")
                    return

                audio, sr = sf.read(wav_path, dtype="float32")

                sd.play(audio, sr)
                sd.wait()

            finally:
                if self.on_done:
                    self.on_done()