import torch
import torch.nn as nn
from timm.layers import LayerNorm2d


# 自定义动态 Tanh 层类
class DynamicTanh(nn.Module):
    """
    动态 Tanh 层，根据输入动态调整 Tanh 的缩放因子 alpha。

    参数:
        normalized_shape (int 或 tuple): 归一化应用的维度形状。
        channels_last (bool): 是否使用通道最后的数据格式（即 NHWC）。
        alpha_init_value (float, 可选): alpha 的初始值，默认为 0.5。
    """
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        # 归一化应用的维度形状
        self.normalized_shape = normalized_shape
        # alpha 的初始值
        self.alpha_init_value = alpha_init_value
        # 是否使用通道最后的数据格式
        self.channels_last = channels_last

        # 初始化 alpha 参数为标量张量，值为 alpha_init_value，并设置为可训练参数
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        # 初始化权重参数为与归一化形状相同的张量，值为 1，并设置为可训练参数
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # 初始化偏置参数为与归一化形状相同的张量，值为 0，并设置为可训练参数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        """
        前向传播方法。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过动态 Tanh 处理后的输出张量。
        """
        # 对输入张量应用 Tanh 函数，缩放因子为 alpha
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            # 如果使用通道最后的数据格式，则对最后一个维度应用权重和偏置
            x = x * self.weight + self.bias
        else:
            # 否则，对第一个维度应用权重和偏置
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        # 返回处理后的张量
        return x

    def extra_repr(self):
        """
        返回对象的额外表示信息，用于打印模型摘要。

        返回:
            str: 包含参数信息的字符串。
        """
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


# 函数：将 LayerNorm 层转换为 DynamicTanh 层
def convert_ln_to_dyt(module):
    """
    将模型中的 LayerNorm 层转换为 DynamicTanh 层。

    参数:
        module (nn.Module): 输入的 PyTorch 模块。

    返回:
        nn.Module: 转换后的 PyTorch 模块。
    """
    # 初始化输出模块为输入模块
    module_output = module

    if isinstance(module, nn.LayerNorm):
        # 如果模块是 LayerNorm，则创建一个 DynamicTanh 层
        # 参数 normalized_shape 继承自 LayerNorm 的 normalized_shape
        # channels_last 根据 LayerNorm 是否为 LayerNorm2d 来确定
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d))

    for name, child in module.named_children():
        # 递归地转换子模块
        module_output.add_module(name, convert_ln_to_dyt(child))
    
    # 删除原始模块
    del module

    # 返回转换后的模块
    return module_output


if __name__ == "__main__":
    # 创建一个示例模型，包含多个 LayerNorm 层
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.LayerNorm(10),
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.LayerNorm2d(16),
        nn.ReLU()
    )

    # 将 LayerNorm 层转换为 DynamicTanh 层
    converted_model = convert_ln_to_dyt(model)

    # 打印转换后的模型结构
    print(converted_model)
