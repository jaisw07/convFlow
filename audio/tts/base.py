from abc import ABC, abstractmethod


class BaseTTS(ABC):
    """
    Abstract base class for all TTS engines.

    Contract:
    - speak(text): start speaking asynchronously
    - stop(): interrupt current speech
    """

    @abstractmethod
    def speak(self, text: str):
        pass

    @abstractmethod
    def stop(self):
        pass