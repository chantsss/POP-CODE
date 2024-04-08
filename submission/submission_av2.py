import osfrom pathlib import Path

import torchfrom av2.datasets.motion_forecasting.eval.submission import ChallengeSubmissionfrom torch import Tensor
from submission.submission_base import SubmissionBase


class SubmissionAv2(SubmissionBase):
    def __init__(self, save_dir: str="", filename: str="", submit=True) -> None:
        super().__init__(save_dir, filename)
        self.submission_file = Path(self.save_dir) / f"{self.filename}.parquet"
        self.challenge_submission = ChallengeSubmission(predictions={})
        self.submit = submit

    def format_data(
        self,
        data: dict,
        trajectory: Tensor,
        probability: Tensor,
        normalized_probability=False,
    ) -> None:
        """
        trajectory: (B, M, 60, 2)
        probability: (B, M)
        normalized_probability: if the input probability is normalized,
        """
        scenario_ids = data["scenario_id"]
        track_ids = data["track_id"]
        batch = len(track_ids)

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

        if not self.submit:
            return global_trajectory, probability

        for i, (scene_id, track_id) in enumerate(zip(scenario_ids, track_ids)):
            # self.challenge_submission.predictions[scene_id] = {
            #     track_id: (global_trajectory[i], probability[i])
            # }
            # global_trajectory_i = {track_id:global_trajectory[i]}
            scene_predictions = {track_id:(global_trajectory[i], probability[i])}
            self.challenge_submission.predictions[scene_id] = scene_predictions
            

    def generate_submission_file(self):
        print(
            "generating submission file for argoverse 2.0 motion forecasting challenge"
        )
        self.challenge_submission.to_parquet(self.submission_file)
        print(f"file saved to {self.submission_file}")

    def submit(self):
        raise NotImplementedError
