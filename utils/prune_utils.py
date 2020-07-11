import torch

def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    return sr

class BNOptimizer():
    @staticmethod
    def updateBN(sr_flag, bn_list, s, epoch):
        if sr_flag:
            for idx in range(len(bn_list)):
                bn = bn_list[idx]
                bn.weight.grad.data.add_(s * torch.sign(bn.weight.data)) # L1

def gather_bn_weights(bn_list):
    size_list = [bn_list[idx].weight.data.shape[0] for idx in range(len(bn_list))]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in enumerate(size_list):
        bn_weights[index:(index + size)] = bn_list[idx].weight.data.abs().clone()
        index += size

    return bn_weights