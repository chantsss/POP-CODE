import abcfrom typing import List


class AugBase:
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def check_data(self, data) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def augment(self, data):
        raise NotImplementedError
