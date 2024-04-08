# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>
"""Classes and utilities used to build submissions for the AV2 motion forecasting challenge."""
from __future__ import annotations
from dataclasses import dataclassfrom pathlib import Pathfrom typing import Dict, Final, List, Tuple

import numpy as np
import pandas as pd
from .constants import AV2_SCENARIO_PRED_TIMESTEPSfrom av2.utils.typing import NDArrayNumber

# Define type aliases used for submission
ScenarioProbabilities = NDArrayNumber  # (K,) per-scenario probabilities, one value for each predicted future.
TrackTrajectories = NDArrayNumber  # (K, AV2_SCENARIO_PRED_TIMESTEPS, 2) per-track predicted trajectories.

ScenarioTrajectories = Dict[str, TrackTrajectories]
ScenarioPredictions = Tuple[ScenarioProbabilities, ScenarioTrajectories]
PredictionRow = Tuple[str, str, float, TrackTrajectories, ScenarioProbabilities]

SUBMISSION_COL_NAMES: Final[List[str]] = [
    "scenario_id",
    "track_id",
    "probability",
    "predicted_trajectory_x",
    "predicted_trajectory_y",
]
EXPECTED_PREDICTION_SHAPE: Final[Tuple[int, ...]] = (AV2_SCENARIO_PRED_TIMESTEPS, 2)


@dataclass(frozen=True)
class ChallengeSubmission:
    """Representation used to build submission for the AV2 motion forecasting challenge.

    Args:
        predictions: Container for all predictions to score - mapping from scenario ID to scenario-level predictions.
    """

    predictions: Dict[str, ScenarioPredictions]

    def __post_init__(self) -> None:
        """Validate that each of the submitted predictions has the appropriate shape and normalized probabilities.

        Raises:
            ValueError: If predictions for at least one track are not of shape (*, AV2_SCENARIO_PRED_TIMESTEPS, 2).
            ValueError: If for any track, number of probabilities doesn't match the number of predicted trajectories.
            ValueError: If prediction probabilities for at least one scenario do not sum to 1.
        """
        for scenario_id, (
            scenario_probabilities,
            scenario_trajectories,
        ) in self.predictions.items():
            for track_id, track_trajectories in scenario_trajectories.items():
                # Validate that predicted trajectories are of the correct shape
                if track_trajectories[0].shape[-2:] != EXPECTED_PREDICTION_SHAPE:
                    raise ValueError(
                        f"Prediction for track {track_id} in {scenario_id} found with invalid shape "
                        f"{track_trajectories.shape}, expected (*, {AV2_SCENARIO_PRED_TIMESTEPS}, 2)."
                    )

                # Validate that the number of predicted trajectories and prediction probabilities matches
                if len(track_trajectories) != len(scenario_probabilities):
                    raise ValueError(
                        f"Prediction for track {track_id} in {scenario_id} has "
                        f"{len(track_trajectories)} predicted trajectories, but "
                        f"{len(scenario_probabilities)} probabilities."
                    )

            # Validate that prediction probabilities for each scenario are normalized
            prediction_probability_sum = np.sum(scenario_probabilities)
            probability_is_normalized = np.isclose(1, prediction_probability_sum)
            if not probability_is_normalized:
                raise ValueError(
                    f"Track probabilities must sum to 1, but probabilities for track {track_id} in {scenario_id} "
                    f"sum up to {prediction_probability_sum}."
                )

    @classmethod
    def from_parquet(cls, submission_file_path: Path) -> ChallengeSubmission:
        """Load challenge submission from serialized parquet representation on disk.

        Args:
            submission_file_path: Path to the serialized submission file (in parquet format).

        Returns:
            Challenge submission object initialized using the loaded data.
        """
        # Load submission data and sort rows by descending probability
        submission_df = pd.read_parquet(submission_file_path)
        submission_df.sort_values(by="probability", inplace=True, ascending=False)

        # From serialized data, build scenario-track mapping for predictions
        submission_dict: Dict[str, ScenarioPredictions] = {}
        for scenario_id, scenario_df in submission_df.groupby("scenario_id"):
            scenario_trajectories: ScenarioTrajectories = {}
            for track_id, track_df in scenario_df.groupby("track_id"):
                predicted_trajectories_x = np.stack(
                    track_df.loc[:, "predicted_trajectory_x"].values.tolist()
                )
                predicted_trajectories_y = np.stack(
                    track_df.loc[:, "predicted_trajectory_y"].values.tolist()
                )
                predicted_trajectories = np.stack(
                    (predicted_trajectories_x, predicted_trajectories_y), axis=-1
                )
                scenario_trajectories[track_id] = predicted_trajectories

            scenario_probabilities = np.array(
                track_df.loc[:, "probability"].values.tolist()
            )
            submission_dict[scenario_id] = (
                scenario_probabilities,
                scenario_trajectories,
            )

        return cls(predictions=submission_dict)

    def to_parquet(self, submission_file_path: Path) -> None:
        """Serialize and save challenge submission on disk using parquet representation.

        Args:
            submission_file_path: Path to the desired location for serialized submission file.
        """
        prediction_rows: List[PredictionRow] = []

        # Build list of rows for the submission dataframe
        for scenario_id, (
            scenario_probabilities,
            scenario_trajectories,
        ) in self.predictions.items():
            for track_id, track_trajectories in scenario_trajectories.items():
                for world_idx in range(len(track_trajectories)):
                    prediction_rows.append(
                        (
                            scenario_id,
                            track_id,
                            scenario_probabilities[world_idx],
                            track_trajectories[world_idx, :, 0],
                            track_trajectories[world_idx, :, 1],
                        )
                    )

        # Build submission dataframe and serialize as parquet file
        submission_df = pd.DataFrame(prediction_rows, columns=SUBMISSION_COL_NAMES)
        submission_df.to_parquet(submission_file_path)
