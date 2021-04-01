import torch


def full_disk_to_tiles(full_disk):
    tiles = []
    for sub_i in range(16):
        x = sub_i // 4
        y = sub_i % 4
        tiles.append(full_disk[x*1024:(x+1)*1024, y*1024:(y+1)*1024])
    return tiles

def tiles_to_full_disk(tiles):
    full_disk = torch.zeros((4096, 4096), requires_grad=False, device='cpu')
    for sub_i in range(16):
        x = sub_i // 4
        y = sub_i % 4
        full_disk[x*1024:(x+1)*1024, y*1024:(y+1)*1024] = tiles[sub_i].squeeze()
    return full_disk

# Convert the model output to a real regressed value image.
def bins_to_output(pred, max_divisor):
    pred = torch.nn.functional.softmax(pred.squeeze(), dim=0)

    # find the max probability bin
    _, max_indices = torch.max(pred, 0)
    max_indices = max_indices.unsqueeze(0)

    # make an ordinal scatter against the one hot args.bins
    max_mask = torch.zeros((80, pred.shape[1], pred.shape[2])).to(0)
    scatter_ones = torch.ones(max_indices.shape).to(0)
    scatter_range = torch.arange(80).unsqueeze(1).unsqueeze(1).float().to(0)

    up_max_indices = (max_indices+1).clamp(0, 80-1)
    down_max_indices = (max_indices-1).clamp(0, 80-1)

    mod_max_indices = max_mask.scatter_(0, max_indices, scatter_ones)
    mod_max_indices = mod_max_indices.scatter_(0, up_max_indices, scatter_ones)
    mod_max_indices = mod_max_indices.scatter_(0, down_max_indices, scatter_ones)

    masked_probabilities = (mod_max_indices * pred)
    normed_probabilities = masked_probabilities / masked_probabilities.sum(dim=0) 
    indices = (normed_probabilities * scatter_range).sum(dim=0)
    pred_im = (indices.float() / ((80-1) / max_divisor)).cpu()
    return pred_im.cpu()
