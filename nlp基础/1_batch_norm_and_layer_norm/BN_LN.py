import torch

def batch_norm(x: torch.Tensor, eps=1e-05) -> torch.Tensor:
    # batch_size, channel, height, weight
    B, C, H, W = x.shape
    alpha = torch.nn.Parameter(torch.ones(C))
    beta = torch.nn.Parameter(torch.zeros(C))
    mean = torch.mean(x, dim=(0, B, C), keepdim=True) # [1, C, 1, 1]
    var = torch.var(x, dim=(0, B, C), keepdim=True, unbiased=False)
    output = (x - mean) / torch.sqrt(var + eps)
    output = output * alpha.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    return output # shape不变
    


def layer_norm(x: torch.Tensor, eps=1e-05) -> torch.Tensor:
    # x.shape: [B, C, T] (batch_size, seq_len, hidden_size)
    alpha = torch.nn.Parameter(torch.ones(x.shape[-1]))
    beta = torch.nn.Parameter(torch.zeros(x.shape[-1]))
    mean = torch.mean(x, dim=-1, keepdim=True)
    # unbiased必须设置为False
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    output = (x-mean) / torch.sqrt(var + eps)
    output = alpha * output + beta
    return output # shape不变

if __name__ == '__main__':

    bn_input = torch.rand((2, 3, 4, 4))
    bn = torch.nn.BatchNorm2d(3)
    print('Torch BN is: ', bn(bn_input))
    print('Our BN is: ', batch_norm(bn_input))

    ln_input = torch.rand((2, 3, 4))
    ln = torch.nn.LayerNorm(4)
    print('Torch LN is: ', ln(ln_input))
    print('Our LN is: ', layer_norm(ln_input))
