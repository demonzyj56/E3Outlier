""" Implements discriminative loss for autoencoder. """
import torch
import torch.nn as nn


class DRAELossAutograd(nn.Module):

    def __init__(self, lamb, size_average=True):
        super(DRAELossAutograd, self).__init__()
        self.lamb = lamb
        self.size_average = size_average  # for compatibility, not used

    def forward(self, inputs, targets):
        err = inputs.sub(targets).pow(2).view(inputs.size(0), -1).sum(dim=1, keepdim=False)
        err_sorted, _ = torch.sort(err)
        total_scatter = err.sub(err.mean()).pow(2).sum()
        regul = 1e6
        obj = None
        for i in range(inputs.size(0)-1):
            err_in = err_sorted[:i+1]
            err_out = err_sorted[i+1:]
            within_scatter = err_in.sub(err_in.mean()).pow(2).sum() + \
                             err_out.sub(err_out.mean()).pow(2).sum()
            h = within_scatter.div(total_scatter)
            if h < regul:
                regul = h
                obj = err_in.mean()

        return obj + self.lamb * regul