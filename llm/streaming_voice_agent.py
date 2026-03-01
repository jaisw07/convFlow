import asyncio
import re
from typing import AsyncGenerator


class SentenceChunker:
    def __init__(
        self,
        min_chars: int = 80,
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

            if len(sentence) >= self.min_chars:
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
    def __init__(self, llm, tts):
        self.llm = llm
        self.tts = tts

        self.tts_queue = asyncio.Queue()

    async def _tts_worker(self):
        while True:
            chunk = await self.tts_queue.get()
            if chunk is None:
                break

            done_event = asyncio.Event()

            def on_done():
                done_event.set()

            self.tts.on_done = on_done
            self.tts.speak(chunk)

            await done_event.wait()
            self.tts_queue.task_done()

    async def handle_turn(self, prompt: str):
        chunker = SentenceChunker()
        tts_task = asyncio.create_task(self._tts_worker())

        async for token in self.llm.stream_response(prompt):
            chunks = await chunker.feed(token)

            for sentence in chunks:
                await self.tts_queue.put(sentence)

        # Flush leftover
        remaining = chunker.flush()
        if remaining:
            await self.tts_queue.put(remaining)

        await self.tts_queue.put(None)
        await tts_task