from abc import ABC, abstractmethod
from typing import Optional, Callable

class BaseTTS(ABC):
    """
    Abstract base class for all TTS engines.

    Contract:
    - speak(text): start speaking asynchronously
    - stop(): interrupt current speech
    """
    def __init__(self):
        self.on_done: Optional[Callable[[], None]] = None

    @abstractmethod
    def speak(self, text: str):
        pass

    @abstractmethod
    def stop(self):
        pass