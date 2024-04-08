import os
import envs.module_path

# Import datasetsfrom envs.directory import ProcessedDatasetDirectoryfrom envs.commonroad.commonroad_prediction import CommonroadDatasetPredictionExtractor

# Import modelsfrom models.interface import PredictionNetworkfrom models.encoders.raster_encoder import RasterEncoderfrom models.encoders.polyline_subgraph import PolylineSubgraphsfrom models.encoders.pgp_encoder import PGPEncoderfrom models.aggregators.concat import Concatfrom models.aggregators.global_attention import GlobalAttentionfrom models.aggregators.goal_conditioned import GoalConditionedfrom models.aggregators.pgp import PGPfrom models.decoders.mtp import MTPfrom models.decoders.multipath import Multipathfrom models.decoders.covernet import CoverNetfrom models.decoders.lvm import LVM

# Import metricsfrom metrics.mtp_loss import MTPLossfrom metrics.min_ade import MinADEKfrom metrics.min_fde import MinFDEKfrom metrics.miss_rate import MissRateKfrom metrics.covernet_loss import CoverNetLossfrom metrics.pi_bc import PiBehaviorCloningfrom metrics.goal_pred_nll import GoalPredictionNLLfrom metrics.full_ade import FullADEfrom metrics.full_fde import FullFDE
from typing import List, Dict, Union

# Datasets
def initialize_dataset(dataset_type: str, encoder_type: str, version: str):
    """
    Helper function to initialize appropriate dataset by dataset type string
    """
    # TODO: Update as we add more dataset
    CONFIG_PATH = {
      'train': "conf/prediction_train_config.yaml",
      'eval': "conf/prediction_eval_config.yaml",
    }

    dataset_classes = {
      'commonroad': CommonroadDatasetPredictionExtractor,
    }

    encoder_mapping = {
        'raster_encoder': 'TODO',
        'polyline_subgraphs': 'vectorize',
        'pgp_encoder': 'graph'
    }

    # @note Instead of read inputs/target from the dataset, 
    # here read from the processed data by default.
    encoder_type = 'hivt' if 'hivt' in encoder_type else encoder_mapping[encoder_type]

    return dataset_classes[dataset_type](
      data_path="",
      config_path=os.path.join(
        os.path.dirname(envs.module_path.__file__), CONFIG_PATH[version]),
        encoder_type=encoder_type,
      save_path=ProcessedDatasetDirectory.get_path(dataset_type, 'prediction')
    )

# Models
def initialize_prediction_network(encoder_type: str, aggregator_type: str, decoder_type: str,
                                  encoder_args: Dict, aggregator_args: Union[Dict, None], decoder_args: Dict):
    """
    Helper function to initialize appropriate encoder, aggegator and decoder models
    """
    encoder = initialize_encoder(encoder_type, encoder_args)
    aggregator = initialize_aggregator(aggregator_type, aggregator_args)
    decoder = initialize_decoder(decoder_type, decoder_args)
    model = PredictionNetwork(encoder, aggregator, decoder)

    return model


def initialize_encoder(encoder_type: str, encoder_args: Dict):
    """
    Initialize appropriate encoder by type.
    """
    # TODO: Update as we add more encoder types
    encoder_mapping = {
        'raster_encoder': RasterEncoder,
        'polyline_subgraphs': PolylineSubgraphs,
        'pgp_encoder': PGPEncoder
    }

    return encoder_mapping[encoder_type](encoder_args)


def initialize_aggregator(aggregator_type: str, aggregator_args: Union[Dict, None]):
    """
    Initialize appropriate aggregator by type.
    """
    # TODO: Update as we add more aggregator types
    aggregator_mapping = {
        'concat': Concat,
        'global_attention': GlobalAttention,
        'gc': GoalConditioned,
        'pgp': PGP
    }

    if aggregator_args:
        return aggregator_mapping[aggregator_type](aggregator_args)
    else:
        return aggregator_mapping[aggregator_type]()


def initialize_decoder(decoder_type: str, decoder_args: Dict):
    """
    Initialize appropriate decoder by type.
    """
    # TODO: Update as we add more decoder types
    decoder_mapping = {
        'mtp': MTP,
        'multipath': Multipath,
        'covernet': CoverNet,
        'lvm': LVM
    }

    return decoder_mapping[decoder_type](decoder_args)


# Metrics
def initialize_metric(metric_type: str, metric_args: Dict = None):
    """
    Initialize appropriate metric by type.
    """
    # TODO: Update as we add more metrics
    metric_mapping = {
        'mtp_loss': MTPLoss,
        'covernet_loss': CoverNetLoss,
        'min_ade_k': MinADEK,
        'min_fde_k': MinFDEK,
        'miss_rate_k': MissRateK,
        'pi_bc': PiBehaviorCloning,
        'goal_pred_nll': GoalPredictionNLL,
        'full_ade_k': FullADE,
        'full_fde_k': FullFDE,
    }

    if metric_args is not None:
        return metric_mapping[metric_type](metric_args)
    else:
        return metric_mapping[metric_type]()
