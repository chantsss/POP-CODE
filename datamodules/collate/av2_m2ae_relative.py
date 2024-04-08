import torchfrom torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    data = {}

    for key in [
        "x",
        "x_padding_mask",
        "x_attr",
        "x_pos",
        "x_scored",
        "lane_positions",
        "lane_padding_mask",
        "lane_pos",
    ]:
        data[key] = torch.cat([b[key] for b in batch], dim=0)

    data["batched_x_scored"] = pad_sequence(
        [b["x_scored"] for b in batch], batch_first=True, padding_value=False
    )

    if batch[0]["y"] is not None:
        data["y"] = torch.cat([b["y"] for b in batch], dim=0)
        data["batched_y"] = pad_sequence([b["y"] for b in batch], batch_first=True)
        data["batched_y_padding_mask"] = pad_sequence(
            [b["y_padding_mask"] for b in batch], batch_first=True, padding_value=True
        )

    num_actors = torch.tensor([b["x"].shape[0] for b in batch]).long()
    num_lanes = torch.tensor([b["lane_positions"].shape[0] for b in batch]).long()
    max_actors = num_actors.max().item()

    num_nodes = num_actors + num_lanes
    offset = torch.cat([torch.zeros(1), num_nodes.cumsum(0)[:-1]], dim=0).long()

    x_index, lane_index, edge_index, actor_batch_index, scene_batch_index = (
        [],
        [],
        [],
        [],
        [],
    )
    for i in range(len(batch)):
        x_index.append(batch[i]["x_index"] + offset[i])
        lane_index.append(batch[i]["lane_index"] + offset[i])
        edge_index.append(batch[i]["edge_index"] + offset[i])
        scene_batch_index.append(torch.ones(num_nodes[i]).long() * i)

        actor_flag = torch.zeros(max_actors, dtype=torch.bool)
        actor_flag[: num_actors[i]] = True
        actor_batch_index.append(actor_flag)

    data["x_index"] = torch.cat(x_index, dim=0)
    data["lane_index"] = torch.cat(lane_index, dim=0)
    data["edge_index"] = torch.cat(edge_index, dim=1)
    data["actor_batch_index"] = torch.cat(actor_batch_index, dim=0)
    data["scene_batch_index"] = torch.cat(scene_batch_index, dim=0)

    data["num_actors"] = num_actors
    data["num_lanes"] = num_lanes

    data["scenario_id"] = [b["scenario_id"] for b in batch]
    data["track_id"] = [b["track_id"] for b in batch]
    data["batch_size"] = len(batch)
    data["max_actors"] = max_actors

    return data
