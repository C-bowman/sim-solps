from abc import ABC, abstractmethod
from sims.interface import SolpsInterface


class Instrument(ABC):
    interface: SolpsInterface

    @abstractmethod
    def update_interface(self, interface: SolpsInterface):
        pass

    @abstractmethod
    def log_likelihood(self) -> float:
        pass