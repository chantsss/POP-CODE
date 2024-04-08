import torchfrom torch.nn.utils.rnn import pad_sequencefrom torch_geometric.data import Batch

# av2
# TemporalDataBatch(x=[916, 50, 2], edge_index=[2, 31682], y=[916, 60, 2], 
#                   positions=[916, 110, 2], num_nodes=916, padding_mask=[916, 110], 
#                   bos_mask=[916, 50], rotate_angles=[916], lane_vectors=[20205, 2], 
#                   is_intersections=[20205], lane_actor_index=[2, 450779], 
#                   lane_actor_vectors=[450779, 2], seq_id=[32], x_type=[916], 
#                   x_category=[916], agent_index=[32], origin=[32, 2], theta=[32], av_index=[32],
#                    city=[32], scenario_id=[32], batch=[916], ptr=[33])
# av1
# TemporalDataBatch(x=[571, 20, 2], edge_index=[2, 11814], 
#                   y=[571, 30, 2], positions=[571, 50, 2], 
#                   num_nodes=571, padding_mask=[571, 50], 
#                   bos_mask=[571, 20], rotate_angles=[571], 
#                   lane_vectors=[23958, 2], is_intersections=[23958], 
#                   turn_directions=[23958], traffic_controls=[23958], 
#                   lane_actor_index=[2, 96883], lane_actor_vectors=[96883, 2], 
#                   seq_id=[32], av_index=[32], agent_index=[32], city=[32], 
#                   origin=[32, 2], theta=[32], batch=[571], ptr=[33])

def collate_fn(batch):


    batch_data = Batch.from_data_list(batch)
    batch_data['turn_directions'] = torch.zeros(batch_data['is_intersections'].shape)
    batch_data['traffic_controls'] = torch.zeros(batch_data['is_intersections'].shape)
    # batch_data["track_id"] = [b["track_id"] for b in batch]


    return batch_data
