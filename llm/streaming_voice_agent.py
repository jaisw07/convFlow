import asyncio
import re
from typing import AsyncGenerator
import numpy as np
from livekit import rtc


class SentenceChunker:
    def __init__(
        self,
        min_chars: int = 35,
        max_chars: int = 250,
    ):
        self.buffer = ""
        self.min_chars = min_chars
        self.max_chars = max_chars
        
    async def feed(self, token: str) -> list[str]:
        """
        Add token and return list of completed chunks.
        """
        self.buffer += token
        chunks = []

        # Emit if sentence-ending punctuation
        while True:
            match = re.search(r"[.!?]\s", self.buffer)
            if not match:
                break

            idx = match.end()
            sentence = self.buffer[:idx].strip()

            if len(sentence) >= self.min_chars or len(self.buffer) >= 40:
                chunks.append(sentence)
                self.buffer = self.buffer[idx:].lstrip()
            else:
                break

        # Safety flush if too large
        if len(self.buffer) > self.max_chars:
            chunks.append(self.buffer.strip())
            self.buffer = ""

        return chunks

    def flush(self) -> str:
        leftover = self.buffer.strip()
        self.buffer = ""
        return leftover
    
class StreamingVoiceAgent:
    def __init__(self, llm, tts, audio_source):
        self.llm = llm
        self.tts = tts
        self.audio_source = audio_source
        self.turn_start = None
        self.first_audio_emitted = None
        self.tts_queue = asyncio.Queue()

    async def _tts_worker(self):
        while True:
            chunk = await self.tts_queue.get()
            if chunk is None:
                self.tts_queue.task_done()
                break

            for audio_chunk in self.tts.synthesize(chunk):
                if not self.first_audio_emitted:
                    now = asyncio.get_event_loop().time()
                    print(f"⏱ First Audio Latency: {now - self.turn_start:.3f}s")
                    self.first_audio_emitted = True

            await self.publish_audio(audio_chunk)
            await asyncio.sleep(0.05)
            self.tts_queue.task_done()

    async def handle_turn(self, prompt: str, stt_done_time: float):
        self.turn_start = stt_done_time
        self.first_audio_emitted = False
        first_token_time = None
        chunker = SentenceChunker()
        tts_task = asyncio.create_task(self._tts_worker())

        async for token in self.llm.stream_response(prompt):
            if first_token_time is None:
                first_token_time = asyncio.get_event_loop().time()
                print(f"⏱ First Token Latency: {first_token_time - self.turn_start:.3f}s")
            chunks = await chunker.feed(token)

            for sentence in chunks:
                await self.tts_queue.put(sentence)

        # Flush leftover
        remaining = chunker.flush()
        if remaining:
            await self.tts_queue.put(remaining)

        await self.tts_queue.put(None)
        await tts_task

    # --------- HELPER ------------

    async def publish_audio(self, audio_chunk):
        # Resample 24kHz → 48kHz (simple upsample)
        audio_48k = self.upsample_linear(audio_chunk)

        # Convert float32 [-1,1] → int16 PCM
        audio_int16 = np.clip(audio_48k, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767).astype(np.int16)

        frame_size = 480  # 10ms @ 48kHz

        total_samples = len(audio_int16)

        for start in range(0, total_samples, frame_size):
            chunk = audio_int16[start:start + frame_size]

            if len(chunk) < frame_size:
                chunk = np.pad(chunk, (0, frame_size - len(chunk)))

            audio_frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=48000,
                num_channels=1,
                samples_per_channel=frame_size,
            )

            await self.audio_source.capture_frame(audio_frame)

    def upsample_linear(self, audio_24k: np.ndarray) -> np.ndarray:
        """
        24kHz → 48kHz using linear interpolation.
        """
        if len(audio_24k) < 2:
            return np.repeat(audio_24k, 2)

        x_old = np.arange(len(audio_24k))
        x_new = np.linspace(0, len(audio_24k) - 1, len(audio_24k) * 2)

        return np.interp(x_new, x_old, audio_24k).astype(np.float32)