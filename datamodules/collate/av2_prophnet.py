import torchfrom torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    data = {}

    for key in ["x", "polylines_features"]:
        data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

    if batch[0]["y"] is not None:
        data["y"] = pad_sequence([b["y"] for b in batch], batch_first=True)

    for key in ["x_padding_mask", "polylines_padding_mask"]:
        data[key] = pad_sequence(
            [b[key] for b in batch], batch_first=True, padding_value=True
        )

    data["x_key_padding_mask"] = data["x_padding_mask"].all(-1)
    data["polylines_key_padding_mask"] = data["polylines_padding_mask"].all(-1)

    data["scenario_id"] = [b["scenario_id"] for b in batch]
    data["track_id"] = [b["track_id"] for b in batch]

    data["origin"] = torch.cat([b["origin"] for b in batch], dim=0)
    data["theta"] = torch.cat([b["theta"] for b in batch])

    return data
