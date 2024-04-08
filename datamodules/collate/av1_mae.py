import torchfrom torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    data = {}

    for key in [
        "x",
        "x_positions",
        "x_centers",
        "x_angles",
        "lane_positions",
        "lane_centers",
        "lane_angles",
        "lane_attr",
    ]:
        data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

    if batch[0]["y"] is not None:
        data["y"] = pad_sequence([b["y"] for b in batch], batch_first=True)

    for key in ["x_padding_mask", "lane_padding_mask"]:
        data[key] = pad_sequence(
            [b[key] for b in batch], batch_first=True, padding_value=True
        )

    data["x_key_padding_mask"] = data["x_padding_mask"].all(-1)
    data["lane_key_padding_mask"] = data["lane_padding_mask"].all(-1)
    data["num_actors"] = (~data["x_key_padding_mask"]).sum(-1)
    data["num_lanes"] = (~data["lane_key_padding_mask"]).sum(-1)

    data["sequence_id"] = torch.Tensor([b["sequence_id"] for b in batch])
    data["track_id"] = [b["track_id"] for b in batch]

    data["origin"] = torch.cat([b["origin"] for b in batch], dim=0)
    data["theta"] = torch.stack([b["theta"] for b in batch])

    return data
