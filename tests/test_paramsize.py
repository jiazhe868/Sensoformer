import torch
import torch.nn as nn
from sensoformer.models.network import Sensoformer
from omegaconf import OmegaConf

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_params(ff_dim):
    # 模拟 Config
    cfg = OmegaConf.create({
        'model': {
            'in_channels': 6,
            'num_scalar_features': 20,
            'embed_dim': 128,
            'heads': 4,
            'layers': 3,
            'dropout': 0.1,
            # 我们将在 Sensoformer.__init__ 中手动注入这个值来测试
        }
    })
    
    # 临时实例化模型 (假设你已经应用了支持 feedforward_dim 的修改)
    # 如果用旧代码，这里需要手动修改 network.py 的默认值来测试
    model = Sensoformer(cfg)
    
    # 强制修改 Transformer 的 FFN 维度 (Hack for calculation without changing class)
    # 注意：这只是为了计算演示，实际必须在 __init__ 里改
    layer = model.event_aggregator.layers[0]
    # 重建 Transformer Layer 以改变维度
    new_layer = nn.TransformerEncoderLayer(
        d_model=128, nhead=4, dim_feedforward=ff_dim, batch_first=True
    )
    # 替换所有层
    model.event_aggregator.layers = nn.ModuleList([new_layer for _ in range(3)])
    
    count = count_parameters(model)
    print(f"Feedforward Dim = {ff_dim:<5} | Total Params: {count/1e6:.2f} M")

if __name__ == "__main__":
    print("-" * 40)
    print("Parameter Count Verification")
    print("-" * 40)
    
    # 1. 你的原始代码情况 (Default)
    check_params(2048)
    
    # 2. 你论文 Table 中的声称
    check_params(256)