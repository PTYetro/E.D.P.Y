import torch

# 加载解密后的模型
decrypted_model_path = 'C:/Users/TCTrolong/Desktop/jiemi/output/ajin_jiemi.pth'
decrypted_state_dict = torch.load(decrypted_model_path, map_location=torch.device('cpu'))

# 检查 state_dict 的结构和内容
def analyze_state_dict(state_dict):
    analysis = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            analysis[key] = {
                "type": "Tensor",
                "shape": value.shape,
                "dtype": value.dtype,
                "mean": value.mean().item(),
                "std": value.std().item(),
            }
        else:
            analysis[key] = {
                "type": type(value).__name__,
                "value": value
            }
    return analysis

# 分析解密的 state_dict
decrypted_analysis = analyze_state_dict(decrypted_state_dict)

# 输出分析结果
for key, info in decrypted_analysis.items():
    print(f"Key: {key}")
    print(f"  Type: {info['type']}")
    if info['type'] == "Tensor":
        print(f"  Shape: {info['shape']}")
        print(f"  Dtype: {info['dtype']}")
        print(f"  Mean: {info['mean']}")
        print(f"  Std: {info['std']}")
    else:
        print(f"  Value: {info['value']}")
    print()
