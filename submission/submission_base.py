import abc
from hydra.utils import to_absolute_pathfrom torch import Tensor


class SubmissionBase:
    def __init__(self, save_dir: str, filename: str) -> None:
        self.save_dir = to_absolute_path(save_dir)
        self.filename = filename

    @abc.abstractmethod
    def format_data(
        self,
        data: dict,
        trajectory: Tensor,
        probability: Tensor,
        normalized_probability=False,
    ) -> None:
        """
        trajectory: (B, M, 30, 2)
        probability: (B, M)
        normalized_probability: if the input probability is normalized,
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_submission_file(self):
        raise NotImplementedError

    def submit(self):
        raise NotImplementedError
