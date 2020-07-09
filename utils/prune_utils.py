import torch

def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    return sr

class BNOptimizer():
    @staticmethod
    def updateBN(sr_flag, module_list, s, prune_idx, epoch):
        if sr_flag:
            for idx in prune_idx:
                bn = module_list[idx]['bn']
                bn.weight.grad.data.add_(s * torch.sign(bn.weight.data)) # L1

                
def gather_bn_weights(module_list, prune_idx):
    size_list = [module_list[idx]['bn'].weight.data.shape[0] for idx in prune_idx]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        bn_weights[index:(index + size)] = module_list[idx]['bn'].weight.data.abs().clone()
        index += size

    return bn_weights