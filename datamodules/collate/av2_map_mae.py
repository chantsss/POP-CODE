from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    data = {}

    for key in ["lane_positions", "lane_centers", "lane_angles"]:
        data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

    for key in ["padding_mask"]:
        data[key] = pad_sequence(
            [b[key] for b in batch], batch_first=True, padding_value=True
        )

    data["key_padding_mask"] = data["padding_mask"].all(-1)

    return data
