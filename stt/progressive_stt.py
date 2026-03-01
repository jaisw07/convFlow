import asyncio
import numpy as np
from typing import List

SAMPLE_RATE = 16000


class ProgressiveSTTController:
    def __init__(self, stt, chunk_sec: int = 25, overlap_sec: int = 2):
        self.stt = stt
        self.chunk_sec = chunk_sec
        self.overlap_sec = overlap_sec
        self._pending_tasks = set()

        self.chunk_size = chunk_sec * SAMPLE_RATE
        self.overlap = overlap_sec * SAMPLE_RATE

        self.partial_transcripts: List[str] = []
        self.processed_samples = 0

    async def maybe_process(self, buffer):
        total_samples = buffer.total_samples
        available = total_samples - self.processed_samples

        if available < self.chunk_size:
            return

        new_audio = buffer.get_audio_from(self.processed_samples)

        if len(new_audio) < self.chunk_size:
            return

        chunk = new_audio[:self.chunk_size]
        self.processed_samples += self.chunk_size - self.overlap

        async def _run():
            text = await self.stt.transcribe_async(chunk)
            if text:
                self.partial_transcripts.append(text)

        task = asyncio.create_task(_run())
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def finalize(self, buffer):
        """
        Called at turn end.
        """
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks)

        full_audio = buffer.get_full_turn_audio()

        remaining_audio = full_audio[self.processed_samples:]

        final_parts = list(self.partial_transcripts)

        if len(remaining_audio) > 0:
            last_text = await self.stt.transcribe_async(remaining_audio)
            if last_text:
                final_parts.append(last_text)

        merged = self._stitch_transcripts(final_parts)

        self.reset()

        return merged.strip()

    def reset(self):
        self.partial_transcripts = []
        self.processed_samples = 0

    def _stitch_transcripts(self, chunks):
        if not chunks:
            return ""

        final_text = chunks[0]

        for next_chunk in chunks[1:]:
            final_text = self._merge_overlap(final_text, next_chunk)

        return final_text

    def _merge_overlap(self, prev_text, next_text):
        prev_words = prev_text.split()
        next_words = next_text.split()

        max_overlap = min(len(prev_words), len(next_words), 30)

        for i in range(max_overlap, 0, -1):
            if prev_words[-i:] == next_words[:i]:
                return " ".join(prev_words + next_words[i:])

        return prev_text + " " + next_text