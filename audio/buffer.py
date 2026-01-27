import numpy as np
from collections import deque


class TurnBuffer:
    """
    Accumulates audio frames for a single user turn.

    Responsibilities:
    - Collect speech audio
    - Track silence duration
    - Truncate to last N seconds
    - Pad audio correctly for Smart Turn
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_turn_seconds: float = 8.0,
        min_speech_seconds: float = 1.5,
        silence_trigger_ms: int = 500,
        frame_duration_ms: int = 40,
    ):
        """
        Args:
            sample_rate: Audio sample rate (16kHz)
            max_turn_seconds: Max audio length Smart Turn supports
            min_speech_seconds: Minimum speech before we consider a valid turn
            silence_trigger_ms: Silence duration before checking Smart Turn
            frame_duration_ms: Frame size in milliseconds
        """

        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_turn_seconds)
        self.min_speech_samples = int(sample_rate * min_speech_seconds)

        self.silence_trigger_frames = int(
            silence_trigger_ms / frame_duration_ms
        )

        self.frames = deque()
        self.total_samples = 0

        self._silent_frames = 0
        self._has_speech = False

    # ---------- Frame ingestion ----------

    def add_speech_frame(self, frame: np.ndarray) -> None:
        """Add a frame that contains speech."""
        self.frames.append(frame)
        self.total_samples += len(frame)
        self._silent_frames = 0
        self._has_speech = True

        self._truncate_if_needed()

    def add_silence_frame(self, frame: np.ndarray) -> None:
        """Add a frame that is silence."""
        self.frames.append(frame)
        self.total_samples += len(frame)
        self._silent_frames += 1

        self._truncate_if_needed()

    # ---------- Turn logic ----------

    def should_check_turn(self) -> bool:
        """
        Returns True if:
        - We've heard speech
        - We've accumulated enough speech
        - We've seen enough silence after speech
        """
        if not self._has_speech:
            return False

        if self.total_samples < self.min_speech_samples:
            return False

        return self._silent_frames >= self.silence_trigger_frames

    def get_audio_for_smart_turn(self) -> np.ndarray:
        """
        Returns audio padded/truncated to EXACTLY max_turn_seconds.
        Padding is added at the BEGINNING (Smart Turn requirement).
        """
        audio = np.concatenate(list(self.frames))

        if len(audio) > self.max_samples:
            audio = audio[-self.max_samples:]

        if len(audio) < self.max_samples:
            pad = np.zeros(self.max_samples - len(audio), dtype=np.float32)
            audio = np.concatenate([pad, audio])

        return audio.astype(np.float32)

    def reset(self) -> None:
        """Clear buffer after a turn completes."""
        self.frames.clear()
        self.total_samples = 0
        self._silent_frames = 0
        self._has_speech = False

    # ---------- Internal helpers ----------

    def _truncate_if_needed(self) -> None:
        """Ensure buffer never exceeds max_samples."""
        while self.total_samples > self.max_samples:
            removed = self.frames.popleft()
            self.total_samples -= len(removed)
            
""" if __name__ == "__main__":
    import time
    from mic_input import MicInput
    from vad import SileroVAD

    buffer = TurnBuffer()
    vad = SileroVAD()
    mic = MicInput()

    print("🎤 Speak for a bit, then pause.")
    print("We will print when TurnBuffer decides to check Smart Turn.\n")

    def on_audio_frame(frame):
        is_speaking = vad.process_frame(frame)

        if is_speaking:
            buffer.add_speech_frame(frame)
        else:
            buffer.add_silence_frame(frame)

        if buffer.should_check_turn():
            audio = buffer.get_audio_for_smart_turn()
            print("🧠 TurnBuffer triggered Smart Turn check")
            print(f"   Audio shape: {audio.shape}")
            print(f"   Duration: {len(audio) / 16000:.2f} seconds\n")

            buffer.reset()

    mic.start(on_audio_frame)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic.stop() """