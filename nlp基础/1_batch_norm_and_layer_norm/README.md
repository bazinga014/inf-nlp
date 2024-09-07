# 1.一句话说BN和LN
- BN是在小批量的数据上对每个特征做归一化（一般来说特征数量为channel数量C）
- LN是在单个样本的feature（即hidden_dim）上做归一化

# 2. 源码
```python
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

```
# 测试题
Q1:x.shape: (2, 3, 4, 4)。对x做batch norm。一共会计算几个均值/方差？
A1：3个，因为batch norm是对batch内的所有数据在特征维度上做归一化，所以每个样本的特征数量就是C，即channel数量，所有要做C=3次

Q2:x.shape: (2, 3, 4)。对x做layer norm。一共会计算几个均值/方差？
A2: 2\*3个=6个。因为LN是对单个样本的feature（即hidden_dim）上做归一化。所以一共有2\*3=6个样本，要做6次。

Q3: Normlization之后为什么还要做一个仿射变换（乘alpha再加上beta）？
A3：因为在Normlization之后，分布为标准正态分布。一般Normlization在激活函数之前，激活函数在0附近基本上为线性变化。失去了激活函数的非线性特性。

Q4：x.shape: (2, 3, 4, 4)。对x做batch norm。alpha和beta的shape是多少？
A4：shape为[3]

Q5：x.shape: (B, T, C)。对x做layer norm。alpha和beta的shape是多少？
A5：shape为[C]


Q6：对于batch norm和layer norm 它们区分训练状态和推理状态吗？
A6：BN区分，LN不区分。
    BN在训练过程中，要在每个batch推断整体的均值和方差，是通过指数平滑来做到的；BN在推理过程中，会用训练最后得到的均值和方差来做BN。
    LN在训练和推理过程中，都是对每个token的feature做LN，所以无需区分

