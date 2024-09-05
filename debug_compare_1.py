import torch

# 加载原始模型
original_model_path = 'C:/Users/TCTrolong/Desktop/compare/ajin.pth'
original_state_dict = torch.load(original_model_path, map_location=torch.device('cpu'))

# 加载解密后的模型
decrypted_model_path = 'C:/Users/TCTrolong/Desktop/compare/666decrypted_model.pth'
decrypted_state_dict = torch.load(decrypted_model_path, map_location=torch.device('cpu'))

# 比较两个模型的参数
def compare_state_dicts(state_dict1, state_dict2):
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    # 检查键的差异
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1

    if only_in_1:
        print(f"在原始模型中但不在解密模型中的键: {only_in_1}")
    if only_in_2:
        print(f"在解密模型中但不在原始模型中的键: {only_in_2}")

    # 检查参数差异
    for key in keys1.intersection(keys2):
        param1 = state_dict1[key]
        param2 = state_dict2[key]

        if isinstance(param1, torch.Tensor) and isinstance(param2, torch.Tensor):
            # 仅比较 Tensor 类型的参数
            if not torch.equal(param1, param2):
                print(f"参数 {key} 不一致.")
                print(f"原始模型参数均值: {param1.mean()}, 解密模型参数均值: {param2.mean()}")
        else:
            # 打印非 Tensor 类型的参数信息
            if type(param1) != type(param2):
                print(f"参数 {key} 类型不一致: 原始类型 {type(param1)}, 解密后类型 {type(param2)}")
            else:
                print(f"参数 {key} 类型一致: {type(param1)}")

# 运行比较函数
compare_state_dicts(original_state_dict, decrypted_state_dict)