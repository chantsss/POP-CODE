from models.encoders.encoder import PredictionEncoder
import torch
import torch.nn as nnfrom torch.nn.utils.rnn import pack_padded_sequencefrom typing import Dictfrom models.encoders.pgp_encoder import PGPEncoder, drop_his_pgp


# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PGPEncoderRecons(PredictionEncoder):

    def __init__(self, args: Dict):
        """
        GRU based encoder from PGP. Lane node features and agent histories encoded using GRUs.
        Additionally, agent-node attention layers infuse each node encoding with nearby agent context.
        Finally GAT layers aggregate local context at each node.

        args to include:

        target_agent_feat_size: int Size of target agent features
        target_agent_emb_size: int Size of target agent embedding
        taret_agent_enc_size: int Size of hidden state of target agent GRU encoder

        node_feat_size: int Size of lane node features
        node_emb_size: int Size of lane node embedding
        node_enc_size: int Size of hidden state of lane node GRU encoder

        nbr_feat_size: int Size of neighboring agent features
        nbr_enb_size: int Size of neighboring agent embeddings
        nbr_enc_size: int Size of hidden state of neighboring agent GRU encoders

        num_gat_layers: int Number of GAT layers to use.
        """

        super().__init__()

        self.init_encoder = PGPEncoder(args)
        self.recons_encoder = PGPEncoder(args)
        self.teacher_model = None
        if 'teacher_model_path' in args:
            self.teacher_model = torch.load(args.teacher_model_path)  
        self.reduce_observation_length = args['reduce_observation_length']

    def forward(self, inputs: Dict) -> Dict:
        """
        Forward pass for PGP encoder
        :param inputs: Dictionary with
            target_agent_representation: torch.Tensor, shape [batch_size, t_h, target_agent_feat_size]
            map_representation: Dict with
                'lane_node_feats': torch.Tensor, shape [batch_size, max_nodes, max_poses, node_feat_size]
                'lane_node_masks': torch.Tensor, shape [batch_size, max_nodes, max_poses, node_feat_size]

                (Optional)
                's_next': Edge look-up table pointing to destination node from source node
                'edge_type': Look-up table with edge type

            surrounding_agent_representation: Dict with
                'vehicles': torch.Tensor, shape [batch_size, max_vehicles, t_h, nbr_feat_size]
                'vehicle_masks': torch.Tensor, shape [batch_size, max_vehicles, t_h, nbr_feat_size]
                'pedestrians': torch.Tensor, shape [batch_size, max_peds, t_h, nbr_feat_size]
                'pedestrian_masks': torch.Tensor, shape [batch_size, max_peds, t_h, nbr_feat_size]
            agent_node_masks:  Dict with
                'vehicles': torch.Tensor, shape [batch_size, max_nodes, max_vehicles]
                'pedestrians': torch.Tensor, shape [batch_size, max_nodes, max_pedestrians]

            Optionally may also include the following if edges are defined for graph traversal
            'init_node': Initial node in the lane graph based on track history.
            'node_seq_gt': Ground truth node sequence for pre-training

        :return:
        """

        encodings = {}
        if self.teacher_model != None:
            encodings['teacher_output'] = self.teacher_model.distill(inputs)
        
        if self.reduce_observation_length:
            target_agent_representation, target_agent_representation_mask = drop_his_pgp(inputs)
            inputs['target_agent_representation'] = target_agent_representation
            inputs['target_agent_representation_mask'] = target_agent_representation_mask

        encodings['init_encodings'] = self.init_encoder(inputs)
        encodings['recons_encodings'] = self.recons_encoder(inputs)


        return encodings

