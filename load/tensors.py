import torch


def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    not_none_batches = [b for b in batch if b is not None]
    data_batch = [b['inp'] for b in not_none_batches]
    if 'lengths' in not_none_batches[0]:
        len_batch = [b['lengths'] for b in not_none_batches]
    else:
        len_batch = [len(b['inp'][0][0]) for b in not_none_batches]

    data_batch_tensor = collate_tensors(data_batch)
    len_batch_tensor = torch.as_tensor(len_batch)
    # un-squeeze for broadcasting
    mask_batch_tensor = lengths_to_mask(
        len_batch_tensor, data_batch_tensor.shape[-1]).unsqueeze(1).unsqueeze(1)

    motion = data_batch_tensor
    cond = {'y': {'mask': mask_batch_tensor, 'lengths': len_batch_tensor}}

    cond['y'].update({'original_motion': data_batch_tensor})

    # For reconstructing the original skeleton
    if 'dist' in not_none_batches[0]:
        dist = [b['dist'] for b in not_none_batches]
        dist = collate_tensors(dist)
        cond['y'].update({'distance': dist})

    if 'labels' in not_none_batches[0]:
        labels_batch = [b['labels'] for b in not_none_batches]
        cond['y'].update({'labels': torch.as_tensor(labels_batch, dtype=torch.float32).squeeze()}) #.unsqueeze(1)})  # remove unsqueeze?

        #print(cond['y']["labels"].shape)
        #exit()

    return motion, cond
