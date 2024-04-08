import osfrom pathlib import Path

import torchfrom torch import Tensor
from src.submission.submission_base import SubmissionBase
from .av2_multiagent_submission_protocol.submission import ChallengeSubmission


class SubmissionAv2MultiAgent(SubmissionBase):
    def __init__(self, save_dir: str = "", filename: str = "", submit=True) -> None:
        super().__init__(save_dir, filename)
        self.submission_file = (
            Path(self.save_dir) / f"multiagent_{self.filename}.parquet"
        )
        self.challenge_submission = ChallengeSubmission(predictions={})
        self.submit = submit

    @torch.no_grad()
    def format_data(
        self,
        data: dict,
        trajectory: Tensor,
        probability: Tensor,
        normalized_probability=False,
    ) -> None:
        """
        trajectory: (B, K, N, 60, 2)
        probability: (B, K)
        normalized_probability: if the input probability is normalized,
        """
        scenario_ids_list = data["scenario_id"]
        track_ids_list = data["track_id"]
        scored_agents_list = data["x_scored"].cpu()
        batch = len(scenario_ids_list)

        origin = data["origin"].view(batch, 1, 1, 1, 2).double()
        theta = data["theta"].double()

        rotate_mat = torch.stack(
            [
                torch.cos(theta),
                torch.sin(theta),
                -torch.sin(theta),
                torch.cos(theta),
            ],
            dim=1,
        ).view(batch, 1, 1, 2, 2)

        global_trajectory = (
            (torch.matmul(trajectory[..., :2].double(), rotate_mat) + origin)
            .cpu()
            .numpy()
        )
        if not normalized_probability:
            probability = torch.softmax(probability.double(), dim=-1)

        probability = probability.cpu().numpy()

        if not self.submit:
            return global_trajectory, probability

        for i, (scene_id, track_ids, scored_agents) in enumerate(
            zip(scenario_ids_list, track_ids_list, scored_agents_list)
        ):
            scored_track_id = [
                track_ids[j] for j in range(len(scored_agents)) if scored_agents[j]
            ]
            scored_trajectory = global_trajectory[i, :, scored_agents]
            scenario_predictions = {
                track_id: trajectory
                for track_id, trajectory in zip(scored_track_id, scored_trajectory)
            }

            self.challenge_submission.predictions[scene_id] = (
                probability[i],
                scenario_predictions,
            )

    def generate_submission_file(self):
        print(
            "generating submission file for argoverse 2.0 motion forecasting challenge"
        )
        self.challenge_submission.to_parquet(self.submission_file)
        print(f"file saved to {self.save_dir}/{self.submission_file}")

    def submit(self):
        raise NotImplementedError
