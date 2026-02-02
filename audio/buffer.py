import numpy as np
from collections import deque


class TurnBuffer:
    """
    Accumulates audio frames for a single user turn.

    Responsibilities:
    - Collect full speech turn (for STT)
    - Maintain rolling 8s window (for Smart Turn)
    - Track silence duration
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_turn_seconds: float = 8.0,
        min_speech_seconds: float = 1.5,
        silence_trigger_ms: int = 500,
        frame_duration_ms: int = 40,
    ):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_turn_seconds)
        self.min_speech_samples = int(sample_rate * min_speech_seconds)

        self.silence_trigger_frames = int(
            silence_trigger_ms / frame_duration_ms
        )

        # Full turn buffer (for STT)
        self.frames = deque()
        self.total_samples = 0

        # Smart Turn rolling buffer (last 8s only)
        self.smart_turn_frames = deque()
        self.smart_turn_samples = 0

        self._silent_frames = 0
        self._has_speech = False

    # ---------- Frame ingestion ----------

    def add_speech_frame(self, frame: np.ndarray) -> None:
        self.frames.append(frame)
        self.total_samples += len(frame)

        self.smart_turn_frames.append(frame)
        self.smart_turn_samples += len(frame)

        self._silent_frames = 0
        self._has_speech = True

        self._truncate_smart_turn_if_needed()

    def add_silence_frame(self, frame: np.ndarray) -> None:
        self.frames.append(frame)
        self.total_samples += len(frame)

        self.smart_turn_frames.append(frame)
        self.smart_turn_samples += len(frame)

        self._silent_frames += 1

        self._truncate_smart_turn_if_needed()

    # ---------- Turn logic ----------

    def should_check_turn(self) -> bool:
        if not self._has_speech:
            return False

        if self.total_samples < self.min_speech_samples:
            return False

        return self._silent_frames >= self.silence_trigger_frames

    # ---------- Audio retrieval ----------

    def get_audio_for_smart_turn(self) -> np.ndarray:
        audio = np.concatenate(list(self.smart_turn_frames))

        if len(audio) > self.max_samples:
            audio = audio[-self.max_samples:]

        if len(audio) < self.max_samples:
            pad = np.zeros(self.max_samples - len(audio), dtype=np.float32)
            audio = np.concatenate([pad, audio])

        return audio.astype(np.float32)

    def get_full_turn_audio(self) -> np.ndarray:
        """Return full turn audio for STT."""
        if not self.frames:
            return np.zeros(0, dtype=np.float32)

        return np.concatenate(list(self.frames)).astype(np.float32)

    # ---------- Reset ----------

    def reset(self) -> None:
        self.frames.clear()
        self.smart_turn_frames.clear()

        self.total_samples = 0
        self.smart_turn_samples = 0

        self._silent_frames = 0
        self._has_speech = False

    # ---------- Internal helpers ----------

    def _truncate_smart_turn_if_needed(self) -> None:
        while self.smart_turn_samples > self.max_samples:
            removed = self.smart_turn_frames.popleft()
            self.smart_turn_samples -= len(removed)
