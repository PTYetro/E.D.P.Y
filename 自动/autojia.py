import os
import datetime
import torch
import io
import numpy as np
import pickle  # 用于序列化和反序列化
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, padding

# 获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 日志文件和信息文件的路径
LOG_FILE = os.path.join(CURRENT_DIR, 'jiami_log.txt')
INFO_FILE = os.path.join(CURRENT_DIR, 'jiami_info.txt')

def write_to_log(content):
    """
    记录运行日志到log.txt文件。
    :param content: 需要记录的内容
    """
    with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
        log_file.write(content + '\n')

def write_to_info(content):
    """
    记录加密信息到info.txt文件。
    :param content: 需要记录的内容
    """
    with open(INFO_FILE, 'a', encoding='utf-8') as info_file:
        info_file.write(content + '\n')

def read_run_count():
    """
    读取当前运行次数，如果没有记录文件则初始化为0。
    :return: 当前运行次数
    """
    try:
        with open(INFO_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return len([line for line in lines if line.startswith('运行次数:')])
    except FileNotFoundError:
        return 0

def update_run_count():
    """
    更新当前运行次数。
    :return: 更新后的运行次数
    """
    run_count = read_run_count() + 1
    return run_count

def encrypt_password(original_password):
    """
    基于初始密码和当前时间生成二次加密密码。
    :param original_password: 用户输入的初始密码
    :return: 二次加密密码和加密时间（整数形式）
    """
    now = datetime.datetime.now()
    time_num = now.strftime('%Y%m%d%H%M')  # 格式化时间为 'YYYYMMDDHHMM'
    time_num = int(time_num)

    # 计算加密相关的数字
    time_numx2 = time_num * 2
    original_password_ascii = ''.join([str(ord(c)) for c in original_password])
    original_password_asciix3 = int(original_password_ascii) * 3
    add_num = time_numx2 + original_password_asciix3

    # 生成字符组
    add_num_str = str(add_num)
    array_partitioning = [int(add_num_str[i:i+2]) for i in range(0, len(add_num_str), 2)][::-1]
    char_group = [chr(x) for x in array_partitioning]

    # 插入字符组并生成二次加密密码
    second_encryption = ""
    remaining_numbers = str(add_num)
    index = 0

    for i in range(len(remaining_numbers)):
        second_encryption += remaining_numbers[i]
        if (i + 1) % (index + 1) == 0 and index < len(char_group):
            second_encryption += char_group[index]
            index += 1

    if index < len(char_group):
        second_encryption += ''.join(char_group[index:])

    return second_encryption, time_num

def encrypt_file_in_memory(plaintext, password):
    """
    使用AES-256对数据进行加密。
    :param plaintext: 明文数据
    :param password: 加密使用的密码
    :return: 加密后的数据
    """
    salt = os.urandom(16)  # 随机生成盐
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = kdf.derive(password.encode())  # 生成密钥

    # 初始化AES加密算法
    iv = os.urandom(16)  # 生成随机初始向量
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # 对数据进行PKCS7填充
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(plaintext) + padder.finalize()

    # 加密数据
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    
    return salt + iv + ciphertext

def complex_encrypt_tensor(tensor, key):
    """
    对张量进行加密，添加噪声并打乱顺序。
    :param tensor: 需要加密的张量
    :param key: 加密使用的key
    :return: 加密后的张量
    """
    tensor_np = tensor.cpu().numpy()
    noise = np.random.normal(0, 0.1, tensor_np.shape)  # 生成随机噪声
    encrypted_tensor = tensor_np + noise * key  # 添加噪声

    # 打乱张量中的元素顺序
    perm = np.random.permutation(encrypted_tensor.size)
    encrypted_tensor = encrypted_tensor.flatten()[perm].reshape(tensor_np.shape)
    
    return torch.tensor(encrypted_tensor, dtype=tensor.dtype, device=tensor.device)

def encrypt_non_tensor(value, password, key):
    """
    对非张量类型参数进行加密。
    :param value: 需要加密的非张量值
    :param password: 用于加密的密码
    :param key: 用于加密的key
    :return: 加密后的非张量数据
    """
    serialized_data = pickle.dumps(value)  # 使用pickle序列化为字节
    
    # 将 key 融入到加密过程中
    key_bytes = int(key * 1000).to_bytes(8, byteorder='big', signed=True)
    serialized_data_with_key = key_bytes + serialized_data  # 将 key 作为数据的一部分
    
    encrypted_data = encrypt_file_in_memory(serialized_data_with_key, password)
    return encrypted_data

def encrypt_model_weights(model_state_dict, key, password):
    """
    对模型的每个权重矩阵和非张量参数进行复杂加密。
    :param model_state_dict: 模型状态字典
    :param key: 用于加密的key
    :param password: 用于加密的密码
    :return: 加密后的模型状态字典和加密信息
    """
    encrypted_state_dict = {}
    encrypted_params_info = []  # 用于记录加密的参数信息
    for name, param in model_state_dict.items():
        if isinstance(param, torch.Tensor):
            # 如果是张量，则进行张量加密
            encrypted_param = complex_encrypt_tensor(param, key)
            encrypted_state_dict[name] = encrypted_param
            encrypted_params_info.append(f"参数 {name} 已加密 - 类型: 张量")
        else:
            # 加密非张量类型参数，并包含 key 的影响
            encrypted_state_dict[name] = encrypt_non_tensor(param, password, key)
            encrypted_params_info.append(f"参数 {name} 已加密 - 类型: {type(param)}")
    return encrypted_state_dict, encrypted_params_info

def encrypt_model(file_path, password, key, output_extension='.enc', output_dir=None):
    """
    加密模型文件并保存加密后的文件。
    :param file_path: 原始模型文件路径
    :param password: 用于加密的密码
    :param key: 用于加密的key
    :param output_extension: 输出文件的扩展名
    :param output_dir: 输出文件的目录
    :return: 加密后文件的路径
    """
    # 加载模型状态字典
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    
    # 加密模型的权重和参数
    encrypted_state_dict, encrypted_params_info = encrypt_model_weights(state_dict, key, password)

    # 将加密后的状态字典写入内存缓冲区
    buffer = io.BytesIO()
    torch.save(encrypted_state_dict, buffer)
    buffer.seek(0)

    # 加密内存缓冲区的内容
    plaintext = buffer.read()
    ciphertext = encrypt_file_in_memory(plaintext, password)

    # 设置输出路径
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    encrypted_file_path = os.path.join(output_dir, base_name + output_extension)
    
    # 保存加密后的文件
    with open(encrypted_file_path, 'wb') as f:
        f.write(ciphertext)

    # 写入日志
    write_to_log(f"加密完成: 原文件: {file_path}, 加密后文件: {encrypted_file_path}")
    write_to_log("\n".join(encrypted_params_info))
    return encrypted_file_path

# 输入和执行部分
original_password = input("请输入模型加密初始密码: ")
key = float(input("请输入矩阵加密的key（浮点数）："))
print("初始密码是：", original_password)

# 生成二次加密后的密码和加密时间
encrypted_password, time_num = encrypt_password(original_password)
print(f"二次加密后的密码: {encrypted_password}")
print(f"加密时间（time_num）: {time_num}")

model_path = 'C:/Users/TCTrolong/Desktop/jiami/input/suannai40k.pth'  # 要加密的模型文件路径
output_path = 'C:/Users/TCTrolong/Desktop/jiami/output'  # 输出目录
password = encrypted_password  # 文件加密的密码

# 写入日志
write_to_log(f"开始加密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")
write_to_log(f"---------------------------------------------------")  # 分隔符

# 执行加密过程
encrypted_file_path = encrypt_model(model_path, password, key, output_extension='.niubi', output_dir=output_path)

# 记录 info.txt 信息
run_count = update_run_count()
info_content = f"运行次数: {run_count}\n文件名: {os.path.basename(model_path)}\n文件路径: {model_path}\n初始密码: {original_password}\n二次加密后的密码: {encrypted_password}\n加密时间: {time_num}\nkey: {key}\n加密后文件的输出路径: {encrypted_file_path}\n--------------------------"
write_to_info(info_content)

print(f"Model encrypted and saved to {encrypted_file_path}")



# import os
# import datetime
# import torch
# import torch.nn as nn
# import io
# import numpy as np
# import pickle  # 新增导入
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# from cryptography.hazmat.backends import default_backend
# from cryptography.hazmat.primitives import hashes, padding

# # 获取当前脚本所在目录
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# LOG_FILE = os.path.join(CURRENT_DIR, 'log.txt')
# INFO_FILE = os.path.join(CURRENT_DIR, 'info.txt')

# def write_to_log(content):
#     """记录运行日志到log.txt文件"""
#     with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
#         log_file.write(content + '\n')

# def write_to_info(content):
#     """记录加密信息到info.txt文件"""
#     with open(INFO_FILE, 'a', encoding='utf-8') as info_file:
#         info_file.write(content + '\n')

# def read_run_count():
#     """读取当前运行次数，如果没有记录文件则初始化为0"""
#     try:
#         with open(INFO_FILE, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
#             return len([line for line in lines if line.startswith('运行次数:')])
#     except FileNotFoundError:
#         return 0

# def update_run_count():
#     """更新当前运行次数"""
#     run_count = read_run_count() + 1
#     return run_count

# # 声音转换模型类定义
# class YourModelClass(torch.nn.Module):
#     def __init__(self):
#         super(YourModelClass, self).__init__()
#         self.weight = nn.Parameter(torch.randn(256, 256))  # 示例权重
#         self.sr = torch.tensor(22050.0, requires_grad=False)  # 采样率
#         self.f0 = torch.tensor(1.0, requires_grad=False)  # 基本频率
#         self.config = {'sample_rate': 22050}  # 配置信息
#         self.version = "1.0.0"  # 模型版本

#     def forward(self, x):
#         x = torch.matmul(x, self.weight)
#         return x

# def encrypt_password(original_password):
#     now = datetime.datetime.now()
#     time_num = now.strftime('%Y%m%d%H%M')  # 格式化时间为 'YYYYMMDDHHMM'
#     time_num = int(time_num)

#     time_numx2 = time_num * 2
#     original_password_ascii = ''.join([str(ord(c)) for c in original_password])
#     original_password_asciix3 = int(original_password_ascii) * 3
#     add_num = time_numx2 + original_password_asciix3

#     add_num_str = str(add_num)
#     array_partitioning = [int(add_num_str[i:i+2]) for i in range(0, len(add_num_str), 2)][::-1]
#     char_group = [chr(x) for x in array_partitioning]

#     second_encryption = ""
#     remaining_numbers = str(add_num)
#     index = 0

#     for i in range(len(remaining_numbers)):
#         second_encryption += remaining_numbers[i]
#         if (i + 1) % (index + 1) == 0 and index < len(char_group):
#             second_encryption += char_group[index]
#             index += 1

#     if index < len(char_group):
#         second_encryption += ''.join(char_group[index:])

#     return second_encryption, time_num

# def complex_encrypt_tensor(tensor, key, seed):
#     """使用随机种子和key对张量进行加密"""
#     np.random.seed(seed)  # 使用固定的随机种子
#     tensor_np = tensor.cpu().numpy()
#     noise = np.random.normal(0, 0.1, tensor_np.shape)
#     encrypted_tensor = tensor_np + noise * key
#     perm = np.random.permutation(encrypted_tensor.size)
#     encrypted_tensor = encrypted_tensor.flatten()[perm].reshape(tensor_np.shape)
    
#     return torch.tensor(encrypted_tensor, dtype=tensor.dtype, device=tensor.device)

# # def encrypt_non_tensor(value, password):
# #     """对非张量参数进行加密"""
# #     serialized_data = pickle.dumps(value)  # 使用pickle序列化为字节
# #     encrypted_data = encrypt_file_in_memory(serialized_data, password)
# #     return encrypted_data

# # def encrypt_model_weights(model_state_dict, key, password, seed):
# #     encrypted_state_dict = {}
# #     encrypted_params_info = []  # 用于记录加密的参数信息
# #     for name, param in model_state_dict.items():
# #         if isinstance(param, torch.Tensor):
# #             encrypted_param = complex_encrypt_tensor(param, key, seed)
# #             encrypted_state_dict[name] = encrypted_param
# #             encrypted_params_info.append(f"参数 {name} 已加密 - 类型: 张量")
# #         else:
# #             # 加密非张量类型参数
# #             encrypted_state_dict[name] = encrypt_non_tensor(param, password)
# #             encrypted_params_info.append(f"参数 {name} 已加密 - 类型: {type(param)}")
# #     return encrypted_state_dict, encrypted_params_info



# def encrypt_file_in_memory(plaintext, password):
#     salt = os.urandom(16)
#     kdf = PBKDF2HMAC(
#         algorithm=hashes.SHA256(),
#         length=32,
#         salt=salt,
#         iterations=100000,
#         backend=default_backend()
#     )
#     key = kdf.derive(password.encode())

#     iv = os.urandom(16)
#     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
#     encryptor = cipher.encryptor()

#     padder = padding.PKCS7(128).padder()
#     padded_data = padder.update(plaintext) + padder.finalize()

#     ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    
#     return salt + iv + ciphertext

# def encrypt_model(file_path, password, key, output_extension='.enc', output_dir=None):
#     state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    
#     # 生成随机种子并加密模型权重
#     seed = np.random.randint(0, 1e6)  # 随机种子
#     encrypted_state_dict, encrypted_params_info = encrypt_model_weights(state_dict, key, password, seed)

#     buffer = io.BytesIO()
#     torch.save(encrypted_state_dict, buffer)
#     buffer.seek(0)

#     plaintext = buffer.read()
    
#     # 追加随机种子到加密文件的开头
#     seed_bytes = seed.to_bytes(4, 'big')
#     ciphertext = encrypt_file_in_memory(seed_bytes + plaintext, password)

#     if output_dir is None:
#         output_dir = os.path.dirname(file_path)
#     else:
#         os.makedirs(output_dir, exist_ok=True)
    
#     base_name = os.path.splitext(os.path.basename(file_path))[0]
#     encrypted_file_path = os.path.join(output_dir, base_name + output_extension)
    
#     with open(encrypted_file_path, 'wb') as f:
#         f.write(ciphertext)

#     write_to_log(f"加密完成: 原文件: {file_path}, 加密后文件: {encrypted_file_path}")
#     write_to_log("\n".join(encrypted_params_info))
#     return encrypted_file_path

# # 输入和执行部分
# original_password = input("请输入模型加密初始密码: ")
# key = float(input("请输入矩阵加密的key（浮点数）："))
# print("初始密码是：", original_password)

# encrypted_password, time_num = encrypt_password(original_password)
# print(f"二次加密后的密码: {encrypted_password}")
# print(f"加密时间（time_num）: {time_num}")

# model_path = 'C:/Users/TCTrolong/Desktop/jiami/input/suannai40k.pth'  # 要加密的模型文件路径
# output_path = 'C:/Users/TCTrolong/Desktop/jiami/output'  # 输出目录
# password = encrypted_password  # 文件加密的密码

# write_to_log(f"开始加密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")
# write_to_log(f"--------------------------")  # 分隔符

# encrypted_file_path = encrypt_model(model_path, password, key, output_extension='.niubi', output_dir=output_path)

# # 记录 info.txt 信息
# run_count = update_run_count()
# info_content = f"运行次数: {run_count}\n文件名: {os.path.basename(model_path)}\n文件路径: {model_path}\n初始密码: {original_password}\n二次加密后的密码: {encrypted_password}\n加密时间: {time_num}\nkey: {key}\n加密后文件的输出路径: {encrypted_file_path}\n--------------------------"
# write_to_info(info_content)

# print(f"Model encrypted and saved to {encrypted_file_path}")



# import os
# import datetime
# import torch
# import torch.nn as nn
# import io
# import numpy as np
# import pickle  # 新增导入
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# from cryptography.hazmat.backends import default_backend
# from cryptography.hazmat.primitives import hashes, padding

# # 获取当前脚本所在目录
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# LOG_FILE = os.path.join(CURRENT_DIR, 'log.txt')
# INFO_FILE = os.path.join(CURRENT_DIR, 'info.txt')

# def write_to_log(content):
#     """记录运行日志到log.txt文件"""
#     with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
#         log_file.write(content + '\n')

# def write_to_info(content):
#     """记录加密信息到info.txt文件"""
#     with open(INFO_FILE, 'a', encoding='utf-8') as info_file:
#         info_file.write(content + '\n')

# def read_run_count():
#     """读取当前运行次数，如果没有记录文件则初始化为0"""
#     try:
#         with open(INFO_FILE, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
#             return len([line for line in lines if line.startswith('运行次数:')])
#     except FileNotFoundError:
#         return 0

# def update_run_count():
#     """更新当前运行次数"""
#     run_count = read_run_count() + 1
#     return run_count

# # 声音转换模型类定义
# class YourModelClass(torch.nn.Module):
#     def __init__(self):
#         super(YourModelClass, self).__init__()
#         self.weight = nn.Parameter(torch.randn(256, 256))  # 示例权重
#         self.sr = torch.tensor(22050.0, requires_grad=False)  # 采样率
#         self.f0 = torch.tensor(1.0, requires_grad=False)  # 基本频率
#         self.config = {'sample_rate': 22050}  # 配置信息
#         self.version = "1.0.0"  # 模型版本

#     def forward(self, x):
#         x = torch.matmul(x, self.weight)
#         return x

# def encrypt_password(original_password):
#     now = datetime.datetime.now()
#     time_num = now.strftime('%Y%m%d%H%M')  # 格式化时间为 'YYYYMMDDHHMM'
#     time_num = int(time_num)

#     time_numx2 = time_num * 2
#     original_password_ascii = ''.join([str(ord(c)) for c in original_password])
#     original_password_asciix3 = int(original_password_ascii) * 3
#     add_num = time_numx2 + original_password_asciix3

#     add_num_str = str(add_num)
#     array_partitioning = [int(add_num_str[i:i+2]) for i in range(0, len(add_num_str), 2)][::-1]
#     char_group = [chr(x) for x in array_partitioning]

#     second_encryption = ""
#     remaining_numbers = str(add_num)
#     index = 0

#     for i in range(len(remaining_numbers)):
#         second_encryption += remaining_numbers[i]
#         if (i + 1) % (index + 1) == 0 and index < len(char_group):
#             second_encryption += char_group[index]
#             index += 1

#     if index < len(char_group):
#         second_encryption += ''.join(char_group[index:])

#     return second_encryption, time_num

# def complex_encrypt_tensor(tensor, key, seed):
#     tensor_np = tensor.cpu().numpy()
#     noise = np.random.normal(0, 0.1, tensor_np.shape)
#     encrypted_tensor = tensor_np + noise * key

#     # 使用固定种子生成随机置换
#     np.random.seed(seed)
#     perm = np.random.permutation(encrypted_tensor.size)
#     encrypted_tensor = encrypted_tensor.flatten()[perm].reshape(tensor_np.shape)
    
#     return torch.tensor(encrypted_tensor, dtype=tensor.dtype, device=tensor.device)

# def encrypt_non_tensor(value, password):
#     """对非张量参数进行加密"""
#     serialized_data = pickle.dumps(value)  # 使用pickle序列化为字节
#     encrypted_data = encrypt_file_in_memory(serialized_data, password)
#     return encrypted_data

# def encrypt_model_weights(model_state_dict, key, password, seed):
#     encrypted_state_dict = {}
#     encrypted_params_info = []  # 用于记录加密的参数信息
#     for name, param in model_state_dict.items():
#         if isinstance(param, torch.Tensor):
#             encrypted_param = complex_encrypt_tensor(param, key, seed)
#             encrypted_state_dict[name] = encrypted_param
#             encrypted_params_info.append(f"参数 {name} 已加密 - 类型: 张量")
#         else:
#             # 加密非张量类型参数
#             encrypted_state_dict[name] = encrypt_non_tensor(param, password)
#             encrypted_params_info.append(f"参数 {name} 已加密 - 类型: {type(param)}")
#     return encrypted_state_dict, encrypted_params_info

# def encrypt_file_in_memory(plaintext, password):
#     salt = os.urandom(16)
#     kdf = PBKDF2HMAC(
#         algorithm=hashes.SHA256(),
#         length=32,
#         salt=salt,
#         iterations=100000,
#         backend=default_backend()
#     )
#     key = kdf.derive(password.encode())

#     iv = os.urandom(16)
#     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
#     encryptor = cipher.encryptor()

#     padder = padding.PKCS7(128).padder()
#     padded_data = padder.update(plaintext) + padder.finalize()

#     ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    
#     return salt + iv + ciphertext

# def encrypt_model(file_path, password, key, output_extension='.enc', output_dir=None):
#     state_dict = torch.load(file_path, map_location=torch.device('cpu'))

#     # 生成一个固定的随机种子
#     seed = np.random.randint(0, 100000)
#     encrypted_state_dict, encrypted_params_info = encrypt_model_weights(state_dict, key, password, seed)

#     buffer = io.BytesIO()
#     torch.save({'state_dict': encrypted_state_dict, 'seed': seed}, buffer)  # 存储加密后的权重和种子
#     buffer.seek(0)

#     plaintext = buffer.read()
#     ciphertext = encrypt_file_in_memory(plaintext, password)

#     if output_dir is None:
#         output_dir = os.path.dirname(file_path)
#     else:
#         os.makedirs(output_dir, exist_ok=True)
    
#     base_name = os.path.splitext(os.path.basename(file_path))[0]
#     encrypted_file_path = os.path.join(output_dir, base_name + output_extension)
    
#     with open(encrypted_file_path, 'wb') as f:
#         f.write(ciphertext)

#     write_to_log(f"加密完成: 原文件: {file_path}, 加密后文件: {encrypted_file_path}")
#     write_to_log("\n".join(encrypted_params_info))
#     return encrypted_file_path

# # 输入和执行部分
# original_password = input("请输入模型加密初始密码: ")
# key = float(input("请输入矩阵加密的key（浮点数）："))
# print("初始密码是：", original_password)

# encrypted_password, time_num = encrypt_password(original_password)
# print(f"二次加密后的密码: {encrypted_password}")
# print(f"加密时间（time_num）: {time_num}")

# model_path = 'C:/Users/TCTrolong/Desktop/jiami/input/suannai40k.pth'  # 要加密的模型文件路径
# output_path = 'C:/Users/TCTrolong/Desktop/jiami/output'  # 输出目录
# password = encrypted_password  # 文件加密的密码

# write_to_log(f"开始加密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")
# write_to_log(f"--------------------------")  # 分隔符

# encrypted_file_path = encrypt_model(model_path, password, key, output_extension='.niubi', output_dir=output_path)

# # 记录 info.txt 信息
# run_count = update_run_count()
# info_content = f"运行次数: {run_count}\n文件名: {os.path.basename(model_path)}\n文件路径: {model_path}\n初始密码: {original_password}\n二次加密后的密码: {encrypted_password}\n加密时间: {time_num}\nkey: {key}\n加密后文件的输出路径: {encrypted_file_path}\n--------------------------"
# write_to_info(info_content)

# print(f"Model encrypted and saved to {encrypted_file_path}")



# import os
# import datetime
# import torch
# import torch.nn as nn
# import io
# import numpy as np
# import pickle  # 新增导入
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# from cryptography.hazmat.backends import default_backend
# from cryptography.hazmat.primitives import hashes, padding

# # 获取当前脚本所在目录
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# LOG_FILE = os.path.join(CURRENT_DIR, 'log.txt')
# INFO_FILE = os.path.join(CURRENT_DIR, 'info.txt')

# def write_to_log(content):
#     """记录运行日志到log.txt文件"""
#     with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
#         log_file.write(content + '\n')

# def write_to_info(content):
#     """记录加密信息到info.txt文件"""
#     with open(INFO_FILE, 'a', encoding='utf-8') as info_file:
#         info_file.write(content + '\n')

# def read_run_count():
#     """读取当前运行次数，如果没有记录文件则初始化为0"""
#     try:
#         with open(INFO_FILE, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
#             return len([line for line in lines if line.startswith('运行次数:')])
#     except FileNotFoundError:
#         return 0

# def update_run_count():
#     """更新当前运行次数"""
#     run_count = read_run_count() + 1
#     return run_count

# # 声音转换模型类定义
# class YourModelClass(torch.nn.Module):
#     def __init__(self):
#         super(YourModelClass, self).__init__()
#         self.weight = nn.Parameter(torch.randn(256, 256))  # 示例权重
#         self.sr = torch.tensor(22050.0, requires_grad=False)  # 采样率
#         self.f0 = torch.tensor(1.0, requires_grad=False)  # 基本频率
#         self.config = {'sample_rate': 22050}  # 配置信息
#         self.version = "1.0.0"  # 模型版本

#     def forward(self, x):
#         x = torch.matmul(x, self.weight)
#         return x

# def encrypt_password(original_password):
#     now = datetime.datetime.now()
#     time_num = now.strftime('%Y%m%d%H%M')  # 格式化时间为 'YYYYMMDDHHMM'
#     time_num = int(time_num)

#     time_numx2 = time_num * 2
#     original_password_ascii = ''.join([str(ord(c)) for c in original_password])
#     original_password_asciix3 = int(original_password_ascii) * 3
#     add_num = time_numx2 + original_password_asciix3

#     add_num_str = str(add_num)
#     array_partitioning = [int(add_num_str[i:i+2]) for i in range(0, len(add_num_str), 2)][::-1]
#     char_group = [chr(x) for x in array_partitioning]

#     second_encryption = ""
#     remaining_numbers = str(add_num)
#     index = 0

#     for i in range(len(remaining_numbers)):
#         second_encryption += remaining_numbers[i]
#         if (i + 1) % (index + 1) == 0 and index < len(char_group):
#             second_encryption += char_group[index]
#             index += 1

#     if index < len(char_group):
#         second_encryption += ''.join(char_group[index:])

#     return second_encryption, time_num

# def complex_encrypt_tensor(tensor, key):
#     tensor_np = tensor.cpu().numpy()
#     noise = np.random.normal(0, 0.1, tensor_np.shape)
#     encrypted_tensor = tensor_np + noise * key
#     perm = np.random.permutation(encrypted_tensor.size)
#     encrypted_tensor = encrypted_tensor.flatten()[perm].reshape(tensor_np.shape)
    
#     return torch.tensor(encrypted_tensor, dtype=tensor.dtype, device=tensor.device)

# def encrypt_non_tensor(value, password):
#     """对非张量参数进行加密"""
#     serialized_data = pickle.dumps(value)  # 使用pickle序列化为字节
#     encrypted_data = encrypt_file_in_memory(serialized_data, password)
#     return encrypted_data

# def encrypt_model_weights(model_state_dict, key, password):
#     encrypted_state_dict = {}
#     encrypted_params_info = []  # 用于记录加密的参数信息
#     for name, param in model_state_dict.items():
#         if isinstance(param, torch.Tensor):
#             encrypted_param = complex_encrypt_tensor(param, key)
#             encrypted_state_dict[name] = encrypted_param
#             encrypted_params_info.append(f"参数 {name} 已加密 - 类型: 张量")
#         else:
#             # 加密非张量类型参数
#             encrypted_state_dict[name] = encrypt_non_tensor(param, password)
#             encrypted_params_info.append(f"参数 {name} 已加密 - 类型: {type(param)}")
#     return encrypted_state_dict, encrypted_params_info

# def encrypt_file_in_memory(plaintext, password):
#     salt = os.urandom(16)
#     kdf = PBKDF2HMAC(
#         algorithm=hashes.SHA256(),
#         length=32,
#         salt=salt,
#         iterations=100000,
#         backend=default_backend()
#     )
#     key = kdf.derive(password.encode())

#     iv = os.urandom(16)
#     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
#     encryptor = cipher.encryptor()

#     padder = padding.PKCS7(128).padder()
#     padded_data = padder.update(plaintext) + padder.finalize()

#     ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    
#     return salt + iv + ciphertext

# def encrypt_model(file_path, password, key, output_extension='.enc', output_dir=None):
#     state_dict = torch.load(file_path, map_location=torch.device('cpu'))
#     encrypted_state_dict, encrypted_params_info = encrypt_model_weights(state_dict, key, password)

#     buffer = io.BytesIO()
#     torch.save(encrypted_state_dict, buffer)
#     buffer.seek(0)

#     plaintext = buffer.read()
#     ciphertext = encrypt_file_in_memory(plaintext, password)

#     if output_dir is None:
#         output_dir = os.path.dirname(file_path)
#     else:
#         os.makedirs(output_dir, exist_ok=True)
    
#     base_name = os.path.splitext(os.path.basename(file_path))[0]
#     encrypted_file_path = os.path.join(output_dir, base_name + output_extension)
    
#     with open(encrypted_file_path, 'wb') as f:
#         f.write(ciphertext)

#     write_to_log(f"加密完成: 原文件: {file_path}, 加密后文件: {encrypted_file_path}")
#     write_to_log("\n".join(encrypted_params_info))
#     return encrypted_file_path

# # 输入和执行部分
# original_password = input("请输入模型加密初始密码: ")
# key = float(input("请输入矩阵加密的key（浮点数）："))
# print("初始密码是：", original_password)

# encrypted_password, time_num = encrypt_password(original_password)
# print(f"二次加密后的密码: {encrypted_password}")
# print(f"加密时间（time_num）: {time_num}")

# model_path = 'C:/Users/TCTrolong/Desktop/jiami/input/suannai40k.pth'  # 要加密的模型文件路径
# output_path = 'C:/Users/TCTrolong/Desktop/jiami/output'  # 输出目录
# password = encrypted_password  # 文件加密的密码

# write_to_log(f"开始加密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")
# write_to_log(f"--------------------------")  # 分隔符

# encrypted_file_path = encrypt_model(model_path, password, key, output_extension='.niubi', output_dir=output_path)

# # 记录 info.txt 信息
# run_count = update_run_count()
# info_content = f"运行次数: {run_count}\n文件名: {os.path.basename(model_path)}\n文件路径: {model_path}\n初始密码: {original_password}\n二次加密后的密码: {encrypted_password}\n加密时间: {time_num}\nkey: {key}\n加密后文件的输出路径: {encrypted_file_path}\n--------------------------"
# write_to_info(info_content)

# print(f"Model encrypted and saved to {encrypted_file_path}")


# import os
# import datetime
# import torch
# import torch.nn as nn
# import io
# import numpy as np
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# from cryptography.hazmat.backends import default_backend
# from cryptography.hazmat.primitives import hashes, padding

# # 获取当前脚本所在的目录路径
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# LOG_FILE = os.path.join(CURRENT_DIR, 'log.txt')
# INFO_FILE = os.path.join(CURRENT_DIR, 'info.txt')

# def write_to_log(content):
#     """记录运行日志到log.txt文件"""
#     with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
#         log_file.write(content + '\n')

# def write_to_info(content):
#     """记录加密信息到info.txt文件"""
#     with open(INFO_FILE, 'a', encoding='utf-8') as info_file:
#         info_file.write(content + '\n')

# def read_run_count():
#     """读取当前运行次数，如果没有记录文件则初始化为0"""
#     try:
#         with open(INFO_FILE, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
#             return len([line for line in lines if line.startswith('运行次数:')])
#     except FileNotFoundError:
#         return 0

# def update_run_count():
#     """更新当前运行次数"""
#     run_count = read_run_count() + 1
#     return run_count

# # 声音转换模型类定义
# class YourModelClass(torch.nn.Module):
#     def __init__(self):
#         super(YourModelClass, self).__init__()
#         self.weight = nn.Parameter(torch.randn(256, 256))  # 示例权重
#         self.sr = torch.tensor(22050.0, requires_grad=False)  # 采样率
#         self.f0 = torch.tensor(1.0, requires_grad=False)  # 基本频率
#         self.config = {'sample_rate': 22050}  # 配置信息
#         self.version = "1.0.0"  # 模型版本

#     def forward(self, x):
#         x = torch.matmul(x, self.weight)
#         return x

# def encrypt_password(original_password):
#     now = datetime.datetime.now()
#     time_num = now.strftime('%Y%m%d%H%M')  # 格式化时间为 'YYYYMMDDHHMM'
#     time_num = int(time_num)

#     time_numx2 = time_num * 2
#     original_password_ascii = ''.join([str(ord(c)) for c in original_password])
#     original_password_asciix3 = int(original_password_ascii) * 3
#     add_num = time_numx2 + original_password_asciix3

#     add_num_str = str(add_num)
#     array_partitioning = [int(add_num_str[i:i+2]) for i in range(0, len(add_num_str), 2)][::-1]
#     char_group = [chr(x) for x in array_partitioning]

#     second_encryption = ""
#     remaining_numbers = str(add_num)
#     index = 0

#     for i in range(len(remaining_numbers)):
#         second_encryption += remaining_numbers[i]
#         if (i + 1) % (index + 1) == 0 and index < len(char_group):
#             second_encryption += char_group[index]
#             index += 1

#     if index < len(char_group):
#         second_encryption += ''.join(char_group[index:])

#     return second_encryption, time_num

# def complex_encrypt_tensor(tensor, key):
#     tensor_np = tensor.cpu().numpy()
#     noise = np.random.normal(0, 0.1, tensor_np.shape)
#     encrypted_tensor = tensor_np + noise * key
#     perm = np.random.permutation(encrypted_tensor.size)
#     encrypted_tensor = encrypted_tensor.flatten()[perm].reshape(tensor_np.shape)
    
#     return torch.tensor(encrypted_tensor, dtype=tensor.dtype, device=tensor.device)

# def encrypt_model_weights(model_state_dict, key):
#     encrypted_state_dict = {}
#     for name, param in model_state_dict.items():
#         if isinstance(param, torch.Tensor):
#             encrypted_param = complex_encrypt_tensor(param, key)
#             encrypted_state_dict[name] = encrypted_param
#             write_to_log(f"加密参数: {name} - 类型: {type(param)}")
#         else:
#             encrypted_state_dict[name] = param
#             write_to_log(f"跳过非张量参数: {name} - 类型: {type(param)}")
#     return encrypted_state_dict

# def encrypt_file_in_memory(plaintext, password):
#     salt = os.urandom(16)
#     kdf = PBKDF2HMAC(
#         algorithm=hashes.SHA256(),
#         length=32,
#         salt=salt,
#         iterations=100000,
#         backend=default_backend()
#     )
#     key = kdf.derive(password.encode())

#     iv = os.urandom(16)
#     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
#     encryptor = cipher.encryptor()

#     padder = padding.PKCS7(128).padder()
#     padded_data = padder.update(plaintext) + padder.finalize()

#     ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    
#     return salt + iv + ciphertext

# def encrypt_model(file_path, password, key, output_extension='.enc', output_dir=None):
#     state_dict = torch.load(file_path, map_location=torch.device('cpu'))
#     encrypted_state_dict = encrypt_model_weights(state_dict, key)

#     buffer = io.BytesIO()
#     torch.save(encrypted_state_dict, buffer)
#     buffer.seek(0)

#     plaintext = buffer.read()
#     ciphertext = encrypt_file_in_memory(plaintext, password)

#     if output_dir is None:
#         output_dir = os.path.dirname(file_path)
#     else:
#         os.makedirs(output_dir, exist_ok=True)
    
#     base_name = os.path.splitext(os.path.basename(file_path))[0]
#     encrypted_file_path = os.path.join(output_dir, base_name + output_extension)
    
#     with open(encrypted_file_path, 'wb') as f:
#         f.write(ciphertext)

#     write_to_log(f"加密完成: 原文件: {file_path}, 加密后文件: {encrypted_file_path}")
#     return encrypted_file_path

# # 输入和执行部分
# original_password = input("请输入模型加密初始密码: ")
# key = float(input("请输入矩阵加密的key（浮点数）："))
# print("初始密码是：", original_password)

# encrypted_password, time_num = encrypt_password(original_password)
# print(f"二次加密后的密码: {encrypted_password}")
# print(f"加密时间（time_num）: {time_num}")

# write_to_log(f"--------------------------\n开始加密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")

# model_path = 'C:/Users/TCTrolong/Desktop/jiami/input/suannai40k.pth'  # 要加密的模型文件路径
# output_path = 'C:/Users/TCTrolong/Desktop/jiami/output'  # 输出目录
# password = encrypted_password  # 文件加密的密码

# encrypted_file_path = encrypt_model(model_path, password, key, output_extension='.niubi', output_dir=output_path)

# # 记录 info.txt 信息
# run_count = update_run_count()
# info_content = f"运行次数: {run_count}\n文件名: {os.path.basename(model_path)}\n文件路径: {model_path}\n初始密码: {original_password}\n二次加密后的密码: {encrypted_password}\n加密时间: {time_num}\nkey: {key}\n加密后文件的输出路径: {encrypted_file_path}\n--------------------------"
# write_to_info(info_content)

# print(f"Model encrypted and saved to {encrypted_file_path}")


# import os
# import datetime
# import torch
# import torch.nn as nn
# import io
# import numpy as np
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# from cryptography.hazmat.backends import default_backend
# from cryptography.hazmat.primitives import hashes, padding

# LOG_FILE = 'log.txt'
# INFO_FILE = 'info.txt'

# def write_to_log(content):
#     """记录运行日志到log.txt文件"""
#     with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
#         log_file.write(content + '\n')

# def write_to_info(content):
#     """记录加密信息到info.txt文件"""
#     with open(INFO_FILE, 'a', encoding='utf-8') as info_file:
#         info_file.write(content + '\n')

# def read_run_count():
#     """读取当前运行次数，如果没有记录文件则初始化为0"""
#     try:
#         with open(INFO_FILE, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
#             return len([line for line in lines if line.startswith('运行次数:')])
#     except FileNotFoundError:
#         return 0

# def update_run_count():
#     """更新当前运行次数"""
#     run_count = read_run_count() + 1
#     return run_count

# # 声音转换模型类定义
# class YourModelClass(torch.nn.Module):
#     def __init__(self):
#         super(YourModelClass, self).__init__()
#         self.weight = nn.Parameter(torch.randn(256, 256))  # 示例权重
#         self.sr = torch.tensor(22050.0, requires_grad=False)  # 采样率
#         self.f0 = torch.tensor(1.0, requires_grad=False)  # 基本频率
#         self.config = {'sample_rate': 22050}  # 配置信息
#         self.version = "1.0.0"  # 模型版本

#     def forward(self, x):
#         x = torch.matmul(x, self.weight)
#         return x

# def encrypt_password(original_password):
#     now = datetime.datetime.now()
#     time_num = now.strftime('%Y%m%d%H%M')  # 格式化时间为 'YYYYMMDDHHMM'
#     time_num = int(time_num)

#     time_numx2 = time_num * 2
#     original_password_ascii = ''.join([str(ord(c)) for c in original_password])
#     original_password_asciix3 = int(original_password_ascii) * 3
#     add_num = time_numx2 + original_password_asciix3

#     add_num_str = str(add_num)
#     array_partitioning = [int(add_num_str[i:i+2]) for i in range(0, len(add_num_str), 2)][::-1]
#     char_group = [chr(x) for x in array_partitioning]

#     second_encryption = ""
#     remaining_numbers = str(add_num)
#     index = 0

#     for i in range(len(remaining_numbers)):
#         second_encryption += remaining_numbers[i]
#         if (i + 1) % (index + 1) == 0 and index < len(char_group):
#             second_encryption += char_group[index]
#             index += 1

#     if index < len(char_group):
#         second_encryption += ''.join(char_group[index:])

#     return second_encryption, time_num

# def complex_encrypt_tensor(tensor, key):
#     tensor_np = tensor.cpu().numpy()
#     noise = np.random.normal(0, 0.1, tensor_np.shape)
#     encrypted_tensor = tensor_np + noise * key
#     perm = np.random.permutation(encrypted_tensor.size)
#     encrypted_tensor = encrypted_tensor.flatten()[perm].reshape(tensor_np.shape)
    
#     return torch.tensor(encrypted_tensor, dtype=tensor.dtype, device=tensor.device)

# def encrypt_model_weights(model_state_dict, key):
#     encrypted_state_dict = {}
#     for name, param in model_state_dict.items():
#         if isinstance(param, torch.Tensor):
#             encrypted_param = complex_encrypt_tensor(param, key)
#             encrypted_state_dict[name] = encrypted_param
#         else:
#             encrypted_state_dict[name] = param
#     return encrypted_state_dict

# def encrypt_file_in_memory(plaintext, password):
#     salt = os.urandom(16)
#     kdf = PBKDF2HMAC(
#         algorithm=hashes.SHA256(),
#         length=32,
#         salt=salt,
#         iterations=100000,
#         backend=default_backend()
#     )
#     key = kdf.derive(password.encode())

#     iv = os.urandom(16)
#     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
#     encryptor = cipher.encryptor()

#     padder = padding.PKCS7(128).padder()
#     padded_data = padder.update(plaintext) + padder.finalize()

#     ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    
#     return salt + iv + ciphertext

# def encrypt_model(file_path, password, key, output_extension='.enc', output_dir=None):
#     state_dict = torch.load(file_path, map_location=torch.device('cpu'))
#     encrypted_state_dict = encrypt_model_weights(state_dict, key)

#     buffer = io.BytesIO()
#     torch.save(encrypted_state_dict, buffer)
#     buffer.seek(0)

#     plaintext = buffer.read()
#     ciphertext = encrypt_file_in_memory(plaintext, password)

#     if output_dir is None:
#         output_dir = os.path.dirname(file_path)
#     else:
#         os.makedirs(output_dir, exist_ok=True)
    
#     base_name = os.path.splitext(os.path.basename(file_path))[0]
#     encrypted_file_path = os.path.join(output_dir, base_name + output_extension)
    
#     with open(encrypted_file_path, 'wb') as f:
#         f.write(ciphertext)

#     write_to_log(f"加密完成: 原文件: {file_path}, 加密后文件: {encrypted_file_path}")
#     return encrypted_file_path

# # 输入和执行部分
# original_password = input("请输入模型加密初始密码: ")
# key = float(input("请输入矩阵加密的key（浮点数）："))
# print("初始密码是：", original_password)

# encrypted_password, time_num = encrypt_password(original_password)
# print(f"二次加密后的密码: {encrypted_password}")
# print(f"加密时间（time_num）: {time_num}")

# model_path = 'C:/Users/TCTrolong/Desktop/jiami/input/suannai40k.pth'  # 要加密的模型文件路径
# output_path = 'C:/Users/TCTrolong/Desktop/jiami/output'  # 输出目录
# password = encrypted_password  # 文件加密的密码

# write_to_log(f"开始加密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")

# encrypted_file_path = encrypt_model(model_path, password, key, output_extension='.niubi', output_dir=output_path)

# # 记录 info.txt 信息
# run_count = update_run_count()
# info_content = f"运行次数: {run_count}\n文件名: {os.path.basename(model_path)}\n文件路径: {model_path}\n初始密码: {original_password}\n二次加密后的密码: {encrypted_password}\n加密时间: {time_num}\nkey: {key}\n加密后文件的输出路径: {encrypted_file_path}\n------"
# write_to_info(info_content)

# print(f"Model encrypted and saved to {encrypted_file_path}")
