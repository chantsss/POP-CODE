import osfrom pathlib import Path

import torchfrom argoverse.evaluation.competition_util import generate_forecasting_h5from torch import Tensor
from submission.submission_base import SubmissionBase


class SubmissionAv1(SubmissionBase):
    def __init__(self, save_dir: str="", filename: str="") -> None:
        super().__init__(save_dir, filename)
        self.h5_file = Path(self.save_dir) / f"{self.filename}.h5"

        self.forecasted_trajectories = {}
        self.forecasted_probabilities = {}

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
        sequence_id = data["seq_id"].cpu().numpy()
        batch = len(sequence_id)

        origin = data["origin"].view(batch, 1, 1, 2).double()
        theta = data["theta"].double()

        rotate_mat = torch.stack(
            [
                torch.cos(theta),
                torch.sin(theta),
                -torch.sin(theta),
                torch.cos(theta),
            ],
            dim=1,
        ).reshape(batch, 2, 2)

        with torch.no_grad():
            global_trajectory = (
                torch.matmul(trajectory[..., :2].double(), rotate_mat.unsqueeze(1))
                + origin
            )
            if not normalized_probability:
                probability = torch.softmax(probability.double(), dim=-1)

        global_trajectory = global_trajectory.detach().cpu().numpy()
        probability = probability.detach().cpu().numpy()

        for i, ID in enumerate(sequence_id):
            self.forecasted_trajectories[ID] = global_trajectory[i]
            self.forecasted_probabilities[ID] = probability[i]

    def generate_submission_file(self):
        print(
            "generating submission file for argoverse 1.0 motion forecasting challenge"
        )
        generate_forecasting_h5(
            data=self.forecasted_trajectories,
            output_path=self.save_dir,
            filename=self.filename,
            probabilities=self.forecasted_probabilities,
        )
        print(f"file saved to {self.save_dir}/{self.filename}")

    def submit(self):
        cmd = f"./scripts/submit_av1.sh {self.h5_file}"
        os.system(cmd)

