import os
import datetime
import torch
import torch.nn as nn
import io
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, padding

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

# 声音转换模型类定义
class YourModelClass(torch.nn.Module):
    def __init__(self):
        super(YourModelClass, self).__init__()
        self.weight = nn.Parameter(torch.randn(256, 256))  # 示例权重
        self.sr = torch.tensor(22050.0, requires_grad=False)  # 采样率
        self.f0 = torch.tensor(1.0, requires_grad=False)  # 基本频率
        self.config = {'sample_rate': 22050}  # 配置信息
        self.version = "1.0.0"  # 模型版本

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        return x
    

    # # 声音转换模型类定义
# class YourModelClass(torch.nn.Module):
#     def __init__(self):
#         super(YourModelClass, self).__init__()
#         # 假设为模型的权重参数，可以根据实际情况调整
#         self.weight = nn.Parameter(torch.randn(256, 256))  # 示例权重
        
#         # 非梯度参数，使用浮点数张量表示，不需要计算梯度
#         self.sr = torch.tensor(22050.0, requires_grad=False)  # 采样率
#         self.f0 = torch.tensor(1.0, requires_grad=False)  # 基本频率
        
#         # 其他模型信息
#         self.config = {'sample_rate': 22050}  # 配置信息
#         self.version = "1.0.0"  # 模型版本

#     def forward(self, x):
#         # 假设模型的前向传播逻辑
#         x = torch.matmul(x, self.weight)
#         return x


def encrypt_password(original_password):
    now = datetime.datetime.now()
    time_num = now.strftime('%Y%m%d%H%M')  # 格式化时间为 'YYYYMMDDHHMM'
    time_num = int(time_num)
    
# def encrypt_password(original_password):
#     # 获取当前时间
#     now = datetime.datetime.now()
#     time_num = now.strftime('%Y%m%d%H%M')  # 格式化时间为 '202409041715'
#     time_num = int(time_num)

    time_numx2 = time_num * 2
    
    original_password_ascii = ''.join([str(ord(c)) for c in original_password])
    original_password_asciix3 = int(original_password_ascii) * 3
    add_num = time_numx2 + original_password_asciix3
    
    add_num_str = str(add_num)
    array_partitioning = [int(add_num_str[i:i+2]) for i in range(0, len(add_num_str), 2)][::-1]
    char_group = [chr(x) for x in array_partitioning]
    
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


# def encrypt_password(original_password):
#     # 获取当前时间
#     now = datetime.datetime.now()
#     time_num = now.strftime('%Y%m%d%H%M')  # 格式化时间为 '202409041715'
#     time_num = int(time_num)
    
#     # 计算 time_num 的两倍
#     time_numx2 = time_num * 2
    
#     # 将初始密码每个字符的 ASCII 转换为字符串
#     original_password_ascii = ''.join([str(ord(c)) for c in original_password])
    
#     # 计算 original_password_ascii 的三倍
#     original_password_asciix3 = int(original_password_ascii) * 3
    
#     # 计算 add_num
#     add_num = time_numx2 + original_password_asciix3
    
#     # 将 add_num 从末尾开始每两位划分为一组，并倒序排列
#     add_num_str = str(add_num)
#     array_partitioning = [int(add_num_str[i:i+2]) for i in range(0, len(add_num_str), 2)][::-1]
    
#     # 对应 ASCII 值转换为字符
#     char_group = [chr(x) for x in array_partitioning]
    
#     # 插入字符组
#     second_encryption = ""
#     remaining_numbers = str(add_num)
#     index = 0

#     for i in range(len(remaining_numbers)):
#         second_encryption += remaining_numbers[i]
#         # 根据位置决定插入字符
#         if (i + 1) % (index + 1) == 0 and index < len(char_group):
#             second_encryption += char_group[index]
#             index += 1

#     # 添加剩余字符（如果有）
#     if index < len(char_group):
#         second_encryption += ''.join(char_group[index:])

#     return second_encryption, time_num  # 返回加密后的密码和时间

def complex_encrypt_tensor(tensor, key):
    """对模型的权重张量进行复杂加密"""
    tensor_np = tensor.cpu().numpy()
    noise = np.random.normal(0, 0.1, tensor_np.shape)
    encrypted_tensor = tensor_np + noise * key
    perm = np.random.permutation(encrypted_tensor.size)
    encrypted_tensor = encrypted_tensor.flatten()[perm].reshape(tensor_np.shape)
    
    return torch.tensor(encrypted_tensor, dtype=tensor.dtype, device=tensor.device)

def encrypt_model_weights(model_state_dict, key):
    """对模型的每个权重张量进行复杂加密"""
    encrypted_state_dict = {}
    for name, param in model_state_dict.items():
        if isinstance(param, torch.Tensor):
            encrypted_param = complex_encrypt_tensor(param, key)
            encrypted_state_dict[name] = encrypted_param
        else:
            encrypted_state_dict[name] = param
    return encrypted_state_dict

# # 复杂的矩阵加密函数
# # def complex_encrypt_matrix(matrix, key):
# #     """对模型的权重矩阵进行复杂加密"""
# #     if isinstance(matrix, torch.Tensor):
# #         matrix_np = matrix.cpu().numpy()

# #         # 非线性扰动
# #         noise = np.random.normal(0, 0.1, matrix_np.shape)
# #         encrypted_matrix = matrix_np + noise * key

# #         # 置换矩阵元素
# #         encrypted_matrix = np.random.permutation(encrypted_matrix.flatten()).reshape(matrix_np.shape)
    
# #         # 返回加密后的矩阵
# #         return torch.tensor(encrypted_matrix, dtype=matrix.dtype, device=matrix.device)
# #     else:
# #         print(f"Skipping non-Tensor type during encryption: {type(matrix)}")
# #         return matrix
# def complex_encrypt_matrix(matrix, key):
#     """对模型的权重矩阵进行复杂加密"""
#     if isinstance(matrix, torch.Tensor):
#         matrix_np = matrix.cpu().numpy()

#         # 非线性扰动
#         noise = np.random.normal(0, 0.1, matrix_np.shape)
#         encrypted_matrix = matrix_np + noise * key

#         # 置换矩阵元素
#         perm = np.random.permutation(encrypted_matrix.size)
#         encrypted_matrix = encrypted_matrix.flatten()[perm].reshape(matrix_np.shape)
    
#         # 返回加密后的矩阵
#         return torch.tensor(encrypted_matrix, dtype=matrix.dtype, device=matrix.device)
#     else:
#         return matrix

# # def encrypt_model_weights(model_state_dict, key):
# #     """对模型的每个权重矩阵进行复杂加密"""
# #     encrypted_state_dict = {}
# #     for name, param in model_state_dict.items():
# #         if isinstance(param, torch.Tensor):
# #             encrypted_param = complex_encrypt_matrix(param, key)
# #             encrypted_state_dict[name] = encrypted_param
# #         elif isinstance(param, dict):  # 如果是嵌套的字典，递归处理
# #             encrypted_state_dict[name] = encrypt_model_weights(param, key)
# #         else:
# #             encrypted_state_dict[name] = param  # 不加密的参数直接保存
# #     return encrypted_state_dict

# def encrypt_model_weights(model_state_dict, key):
#     """对模型的每个权重矩阵进行复杂加密"""
#     encrypted_state_dict = {}
#     for name, param in model_state_dict.items():
#         if isinstance(param, torch.Tensor):
#             encrypted_param = complex_encrypt_matrix(param, key)
#             encrypted_state_dict[name] = encrypted_param
#         elif isinstance(param, dict):  # 递归处理嵌套的字典
#             encrypted_state_dict[name] = encrypt_model_weights(param, key)
#         else:
#             encrypted_state_dict[name] = param
#     return encrypted_state_dict




def encrypt_file_in_memory(plaintext, password):
    salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = kdf.derive(password.encode())

    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(plaintext) + padder.finalize()

    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    
    return salt + iv + ciphertext

# # 文件加密函数（在内存中进行加密）
# # def encrypt_file_in_memory(plaintext, password):
# #     """对文件内容进行 AES-256 加密，避免临时文件"""
# #     salt = os.urandom(16)
# #     kdf = PBKDF2HMAC(
# #         algorithm=hashes.SHA256(),
# #         length=32,
# #         salt=salt,
# #         iterations=100000,
# #         backend=default_backend()
# #     )
# #     key = kdf.derive(password.encode())

# #     # 初始化 AES-256 加密算法
# #     iv = os.urandom(16)
# #     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
# #     encryptor = cipher.encryptor()

# #     # 对数据进行填充
# #     padder = padding.PKCS7(128).padder()
# #     padded_data = padder.update(plaintext) + padder.finalize()

# #     # 加密数据
# #     ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    
# #     print("AES加密成功，密文长度:", len(ciphertext))  # 调试输出，确认密文长度

# #     return salt + iv + ciphertext

# # AES加密函数
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

# # # 综合加密流程（避免临时文件）
# # def encrypt_model(file_path, password, key, output_extension='.enc', output_dir=None):
# #     # 读取并加载模型状态字典
# #     state_dict = torch.load(file_path, map_location=torch.device('cpu'))  # 加载到 CPU
    
# #     # 确认加载的 state_dict 中的内容
# #     print("Loaded state_dict keys:")
# #     for k, v in state_dict.items():
# #         print(f"Key: {k}, Type: {type(v)}")
    
# #     # 对模型权重进行复杂加密
# #     encrypted_state_dict = encrypt_model_weights(state_dict, key)

# #     # 将加密后的 state_dict 存储到内存中的字节流
# #     buffer = io.BytesIO()
# #     torch.save(encrypted_state_dict, buffer)
# #     buffer.seek(0)  # 重置字节流的位置

# #     # 读取内存中的加密模型数据
# #     plaintext = buffer.read()
    
# #     # 确保数据大小不为零
# #     if len(plaintext) == 0:
# #         print("Error: No data to encrypt.")
# #         return

# #     print("模型数据已序列化，长度:", len(plaintext))  # 调试输出，确认数据长度

# #     # 在内存中对模型文件进行 AES-256 加密
# #     ciphertext = encrypt_file_in_memory(plaintext, password)

# #     # 确定加密文件的路径
# #     if output_dir is None:
# #         output_dir = os.path.dirname(file_path)
# #     else:
# #         os.makedirs(output_dir, exist_ok=True)
    
# #     base_name = os.path.splitext(os.path.basename(file_path))[0]
# #     encrypted_file_path = os.path.join(output_dir, base_name + output_extension)
    
# #     # 保存最终加密后的文件
# #     with open(encrypted_file_path, 'wb') as f:
# #         f.write(ciphertext)

# #     print(f"Model encrypted and saved to {encrypted_file_path}")
# #     return encrypted_file_path


def encrypt_model(file_path, password, key, output_extension='.enc', output_dir=None):
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    encrypted_state_dict = encrypt_model_weights(state_dict, key)

    buffer = io.BytesIO()
    torch.save(encrypted_state_dict, buffer)
    buffer.seek(0)

    plaintext = buffer.read()
    ciphertext = encrypt_file_in_memory(plaintext, password)

    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    encrypted_file_path = os.path.join(output_dir, base_name + output_extension)
    
    with open(encrypted_file_path, 'wb') as f:
        f.write(ciphertext)

    print(f"Model encrypted and saved to {encrypted_file_path}")
    return encrypted_file_path

# 输入和执行部分
original_password = input("请输入模型加密初始密码: ")
key = float(input("请输入矩阵加密的key（浮点数）："))
print("初始密码是：", original_password)

encrypted_password, time_num = encrypt_password(original_password)
print(f"二次加密后的密码: {encrypted_password}")
print(f"加密时间（time_num）: {time_num}")

model_path = 'C:/Users/TCTrolong/Desktop/jiami/input/suannai40k.pth'  # 要加密的模型文件路径
output_path = 'C:/Users/TCTrolong/Desktop/jiami/output'  # 输出目录
password = encrypted_password  # 文件加密的密码

encrypt_model(model_path, password, key, output_extension='.niubi', output_dir=output_path)


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

# # 声音转换模型类定义
# class YourModelClass(torch.nn.Module):
#     def __init__(self):
#         super(YourModelClass, self).__init__()
#         # 假设为模型的权重参数，可以根据实际情况调整
#         self.weight = nn.Parameter(torch.randn(256, 256))  # 示例权重
        
#         # 非梯度参数，使用浮点数张量表示，不需要计算梯度
#         self.sr = torch.tensor(22050.0, requires_grad=False)  # 采样率
#         self.f0 = torch.tensor(1.0, requires_grad=False)  # 基本频率
        
#         # 其他模型信息
#         self.config = {'sample_rate': 22050}  # 配置信息
#         self.version = "1.0.0"  # 模型版本

#     def forward(self, x):
#         # 假设模型的前向传播逻辑
#         x = torch.matmul(x, self.weight)
#         return x

# def encrypt_password(original_password):
#     # 获取当前时间
#     now = datetime.datetime.now()
#     time_num = now.strftime('%Y%m%d%H%M')  # 格式化时间为 '202409041715'
#     time_num = int(time_num)
    
#     # 计算 time_num 的两倍
#     time_numx2 = time_num * 2
    
#     # 将初始密码每个字符的 ASCII 转换为字符串
#     original_password_ascii = ''.join([str(ord(c)) for c in original_password])
    
#     # 计算 original_password_ascii 的三倍
#     original_password_asciix3 = int(original_password_ascii) * 3
    
#     # 计算 add_num
#     add_num = time_numx2 + original_password_asciix3
    
#     # 将 add_num 从末尾开始每两位划分为一组，并倒序排列
#     add_num_str = str(add_num)
#     array_partitioning = [int(add_num_str[i:i+2]) for i in range(0, len(add_num_str), 2)][::-1]
    
#     # 对应 ASCII 值转换为字符
#     char_group = [chr(x) for x in array_partitioning]
    
#     # 插入字符组
#     second_encryption = ""
#     remaining_numbers = str(add_num)
#     index = 0

#     for i in range(len(remaining_numbers)):
#         second_encryption += remaining_numbers[i]
#         # 根据位置决定插入字符
#         if (i + 1) % (index + 1) == 0 and index < len(char_group):
#             second_encryption += char_group[index]
#             index += 1

#     # 添加剩余字符（如果有）
#     if index < len(char_group):
#         second_encryption += ''.join(char_group[index:])

#     return second_encryption, time_num  # 返回加密后的密码和时间

# # 复杂的矩阵加密函数
# # def complex_encrypt_matrix(matrix, key):
# #     """对模型的权重矩阵进行复杂加密"""
# #     if isinstance(matrix, torch.Tensor):
# #         matrix_np = matrix.cpu().numpy()

# #         # 非线性扰动
# #         noise = np.random.normal(0, 0.1, matrix_np.shape)
# #         encrypted_matrix = matrix_np + noise * key

# #         # 置换矩阵元素
# #         encrypted_matrix = np.random.permutation(encrypted_matrix.flatten()).reshape(matrix_np.shape)
    
# #         # 返回加密后的矩阵
# #         return torch.tensor(encrypted_matrix, dtype=matrix.dtype, device=matrix.device)
# #     else:
# #         print(f"Skipping non-Tensor type during encryption: {type(matrix)}")
# #         return matrix
# def complex_encrypt_matrix(matrix, key):
#     """对模型的权重矩阵进行复杂加密"""
#     if isinstance(matrix, torch.Tensor):
#         matrix_np = matrix.cpu().numpy()

#         # 非线性扰动
#         noise = np.random.normal(0, 0.1, matrix_np.shape)
#         encrypted_matrix = matrix_np + noise * key

#         # 置换矩阵元素
#         perm = np.random.permutation(encrypted_matrix.size)
#         encrypted_matrix = encrypted_matrix.flatten()[perm].reshape(matrix_np.shape)
    
#         # 返回加密后的矩阵
#         return torch.tensor(encrypted_matrix, dtype=matrix.dtype, device=matrix.device)
#     else:
#         return matrix

# # def encrypt_model_weights(model_state_dict, key):
# #     """对模型的每个权重矩阵进行复杂加密"""
# #     encrypted_state_dict = {}
# #     for name, param in model_state_dict.items():
# #         if isinstance(param, torch.Tensor):
# #             encrypted_param = complex_encrypt_matrix(param, key)
# #             encrypted_state_dict[name] = encrypted_param
# #         elif isinstance(param, dict):  # 如果是嵌套的字典，递归处理
# #             encrypted_state_dict[name] = encrypt_model_weights(param, key)
# #         else:
# #             encrypted_state_dict[name] = param  # 不加密的参数直接保存
# #     return encrypted_state_dict

# def encrypt_model_weights(model_state_dict, key):
#     """对模型的每个权重矩阵进行复杂加密"""
#     encrypted_state_dict = {}
#     for name, param in model_state_dict.items():
#         if isinstance(param, torch.Tensor):
#             encrypted_param = complex_encrypt_matrix(param, key)
#             encrypted_state_dict[name] = encrypted_param
#         elif isinstance(param, dict):  # 递归处理嵌套的字典
#             encrypted_state_dict[name] = encrypt_model_weights(param, key)
#         else:
#             encrypted_state_dict[name] = param
#     return encrypted_state_dict


# # 文件加密函数（在内存中进行加密）
# # def encrypt_file_in_memory(plaintext, password):
# #     """对文件内容进行 AES-256 加密，避免临时文件"""
# #     salt = os.urandom(16)
# #     kdf = PBKDF2HMAC(
# #         algorithm=hashes.SHA256(),
# #         length=32,
# #         salt=salt,
# #         iterations=100000,
# #         backend=default_backend()
# #     )
# #     key = kdf.derive(password.encode())

# #     # 初始化 AES-256 加密算法
# #     iv = os.urandom(16)
# #     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
# #     encryptor = cipher.encryptor()

# #     # 对数据进行填充
# #     padder = padding.PKCS7(128).padder()
# #     padded_data = padder.update(plaintext) + padder.finalize()

# #     # 加密数据
# #     ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    
# #     print("AES加密成功，密文长度:", len(ciphertext))  # 调试输出，确认密文长度

# #     return salt + iv + ciphertext

# # AES加密函数
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

# # # 综合加密流程（避免临时文件）
# # def encrypt_model(file_path, password, key, output_extension='.enc', output_dir=None):
# #     # 读取并加载模型状态字典
# #     state_dict = torch.load(file_path, map_location=torch.device('cpu'))  # 加载到 CPU
    
# #     # 确认加载的 state_dict 中的内容
# #     print("Loaded state_dict keys:")
# #     for k, v in state_dict.items():
# #         print(f"Key: {k}, Type: {type(v)}")
    
# #     # 对模型权重进行复杂加密
# #     encrypted_state_dict = encrypt_model_weights(state_dict, key)

# #     # 将加密后的 state_dict 存储到内存中的字节流
# #     buffer = io.BytesIO()
# #     torch.save(encrypted_state_dict, buffer)
# #     buffer.seek(0)  # 重置字节流的位置

# #     # 读取内存中的加密模型数据
# #     plaintext = buffer.read()
    
# #     # 确保数据大小不为零
# #     if len(plaintext) == 0:
# #         print("Error: No data to encrypt.")
# #         return

# #     print("模型数据已序列化，长度:", len(plaintext))  # 调试输出，确认数据长度

# #     # 在内存中对模型文件进行 AES-256 加密
# #     ciphertext = encrypt_file_in_memory(plaintext, password)

# #     # 确定加密文件的路径
# #     if output_dir is None:
# #         output_dir = os.path.dirname(file_path)
# #     else:
# #         os.makedirs(output_dir, exist_ok=True)
    
# #     base_name = os.path.splitext(os.path.basename(file_path))[0]
# #     encrypted_file_path = os.path.join(output_dir, base_name + output_extension)
    
# #     # 保存最终加密后的文件
# #     with open(encrypted_file_path, 'wb') as f:
# #         f.write(ciphertext)

# #     print(f"Model encrypted and saved to {encrypted_file_path}")
# #     return encrypted_file_path

# # 加密模型函数
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

#     print(f"Model encrypted and saved to {encrypted_file_path}")
#     return encrypted_file_path


# # 输入和执行部分
# original_password = input("请输入模型加密初始密码: ")
# key = float(input("请输入矩阵加密的key（浮点数）："))
# print("初始密码是：", original_password)

# encrypted_password, time_num = encrypt_password(original_password)
# print(f"二次加密后的密码: {encrypted_password}")
# print(f"加密时间（time_num）: {time_num}")

# # 使用
# model_path = 'C:/Users/TCTrolong/Desktop/jiami/input/ajin.pth'  # 要加密的模型文件路径
# output_path = 'C:/Users/TCTrolong/Desktop/jiami/output'  # 输出目录
# password = encrypted_password  # 文件加密的密码

# encrypt_model(model_path, password, key, output_extension='.niubi', output_dir=output_path)


# # import os
# # import datetime
# # import torch
# # import torch.nn as nn
# # import io
# # import numpy as np
# # from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# # from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# # from cryptography.hazmat.backends import default_backend
# # from cryptography.hazmat.primitives import hashes, padding

# # # 声音转换模型类定义
# # class YourModelClass(torch.nn.Module):
# #     def __init__(self):
# #         super(YourModelClass, self).__init__()
# #         # 假设为模型的权重参数，可以根据实际情况调整
# #         self.weight = nn.Parameter(torch.randn(256, 256))  # 示例权重
        
# #         # 非梯度参数，使用浮点数张量表示，不需要计算梯度
# #         self.sr = torch.tensor(22050.0, requires_grad=False)  # 采样率
# #         self.f0 = torch.tensor(1.0, requires_grad=False)  # 基本频率
        
# #         # 其他模型信息
# #         self.config = {'sample_rate': 22050}  # 配置信息
# #         self.version = "1.0.0"  # 模型版本

# #     def forward(self, x):
# #         # 假设模型的前向传播逻辑
# #         x = torch.matmul(x, self.weight)
# #         return x

# # def encrypt_password(original_password):
# #     # 获取当前时间
# #     now = datetime.datetime.now()
# #     time_num = now.strftime('%Y%m%d%H%M')  # 格式化时间为 '202409041715'
# #     time_num = int(time_num)
    
# #     # 计算 time_num 的两倍
# #     time_numx2 = time_num * 2
    
# #     # 将初始密码每个字符的 ASCII 转换为字符串
# #     original_password_ascii = ''.join([str(ord(c)) for c in original_password])
    
# #     # 计算 original_password_ascii 的三倍
# #     original_password_asciix3 = int(original_password_ascii) * 3
    
# #     # 计算 add_num
# #     add_num = time_numx2 + original_password_asciix3
    
# #     # 将 add_num 从末尾开始每两位划分为一组，并倒序排列
# #     add_num_str = str(add_num)
# #     array_partitioning = [int(add_num_str[i:i+2]) for i in range(0, len(add_num_str), 2)][::-1]
    
# #     # 对应 ASCII 值转换为字符
# #     char_group = [chr(x) for x in array_partitioning]
    
# #     # 插入字符组
# #     second_encryption = ""
# #     remaining_numbers = str(add_num)
# #     index = 0

# #     for i in range(len(remaining_numbers)):
# #         second_encryption += remaining_numbers[i]
# #         # 根据位置决定插入字符
# #         if (i + 1) % (index + 1) == 0 and index < len(char_group):
# #             second_encryption += char_group[index]
# #             index += 1

# #     # 添加剩余字符（如果有）
# #     if index < len(char_group):
# #         second_encryption += ''.join(char_group[index:])

# #     return second_encryption, time_num  # 返回加密后的密码和时间

# # # 复杂的矩阵加密函数
# # # def complex_encrypt_matrix(matrix, key):
# # #     """对模型的权重矩阵进行复杂加密"""
# # #     matrix_np = matrix.cpu().numpy()

# # #     # 非线性扰动
# # #     noise = np.random.normal(0, 0.1, matrix_np.shape)
# # #     encrypted_matrix = matrix_np + noise * key

# # #     # 置换矩阵元素
# # #     encrypted_matrix = np.random.permutation(encrypted_matrix.flatten()).reshape(matrix_np.shape)
    
# # #     # 返回加密后的矩阵
# # #     return torch.tensor(encrypted_matrix, dtype=matrix.dtype, device=matrix.device)
# # def complex_encrypt_matrix(matrix, key):
# #     """对模型的权重矩阵进行复杂加密"""
# #     if isinstance(matrix, torch.Tensor):
# #         matrix_np = matrix.cpu().numpy()

# #         # 非线性扰动
# #         noise = np.random.normal(0, 0.1, matrix_np.shape)
# #         encrypted_matrix = matrix_np + noise * key

# #         # 置换矩阵元素
# #         encrypted_matrix = np.random.permutation(encrypted_matrix.flatten()).reshape(matrix_np.shape)
    
# #         # 返回加密后的矩阵
# #         return torch.tensor(encrypted_matrix, dtype=matrix.dtype, device=matrix.device)
# #     else:
# #         # 确保非 Tensor 类型的对象不被加密为 dict
# #         return matrix



# # def encrypt_model_weights(model_state_dict, key):
# #     """对模型的每个权重矩阵进行复杂加密"""
# #     encrypted_state_dict = {}
# #     for name, param in model_state_dict.items():
# #         if isinstance(param, torch.Tensor):
# #             encrypted_param = complex_encrypt_matrix(param, key)
# #             encrypted_state_dict[name] = encrypted_param
# #         elif isinstance(param, dict):  # 如果是嵌套的字典，递归处理
# #             encrypted_state_dict[name] = encrypt_model_weights(param, key)
# #         else:
# #             encrypted_state_dict[name] = param  # 不加密的参数直接保存
# #     return encrypted_state_dict

# # # 文件加密函数（在内存中进行加密）
# # def encrypt_file_in_memory(plaintext, password):
# #     """对文件内容进行 AES-256 加密，避免临时文件"""
# #     salt = os.urandom(16)
# #     kdf = PBKDF2HMAC(
# #         algorithm=hashes.SHA256(),
# #         length=32,
# #         salt=salt,
# #         iterations=100000,
# #         backend=default_backend()
# #     )
# #     key = kdf.derive(password.encode())

# #     # 初始化 AES-256 加密算法
# #     iv = os.urandom(16)
# #     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
# #     encryptor = cipher.encryptor()

# #     # 对数据进行填充
# #     padder = padding.PKCS7(128).padder()
# #     padded_data = padder.update(plaintext) + padder.finalize()

# #     # 加密数据
# #     ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    
# #     print("AES加密成功，密文长度:", len(ciphertext))  # 调试输出，确认密文长度

# #     return salt + iv + ciphertext

# # # 综合加密流程（避免临时文件）
# # def encrypt_model(file_path, password, key, output_extension='.enc', output_dir=None):
# #     # 读取并加载模型状态字典
# #     state_dict = torch.load(file_path, map_location=torch.device('cpu'))  # 加载到 CPU
    
# #     # 确认加载的 state_dict 中的内容
# #     print("Loaded state_dict keys:")
# #     for k, v in state_dict.items():
# #         print(f"Key: {k}, Type: {type(v)}")
    
# #     # 对模型权重进行复杂加密
# #     encrypted_state_dict = encrypt_model_weights(state_dict, key)

# #     # 将加密后的 state_dict 存储到内存中的字节流
# #     buffer = io.BytesIO()
# #     torch.save(encrypted_state_dict, buffer)
# #     buffer.seek(0)  # 重置字节流的位置

# #     # 读取内存中的加密模型数据
# #     plaintext = buffer.read()
    
# #     # 确保数据大小不为零
# #     if len(plaintext) == 0:
# #         print("Error: No data to encrypt.")
# #         return

# #     print("模型数据已序列化，长度:", len(plaintext))  # 调试输出，确认数据长度

# #     # 在内存中对模型文件进行 AES-256 加密
# #     ciphertext = encrypt_file_in_memory(plaintext, password)

# #     # 确定加密文件的路径
# #     if output_dir is None:
# #         output_dir = os.path.dirname(file_path)
# #     else:
# #         os.makedirs(output_dir, exist_ok=True)
    
# #     base_name = os.path.splitext(os.path.basename(file_path))[0]
# #     encrypted_file_path = os.path.join(output_dir, base_name + output_extension)
    
# #     # 保存最终加密后的文件
# #     with open(encrypted_file_path, 'wb') as f:
# #         f.write(ciphertext)

# #     print(f"Model encrypted and saved to {encrypted_file_path}")
# #     return encrypted_file_path

# # # 输入和执行部分
# # original_password = input("请输入模型加密初始密码: ")
# # key = float(input("请输入矩阵加密的key（浮点数）："))
# # print("初始密码是：", original_password)

# # encrypted_password, time_num = encrypt_password(original_password)
# # print(f"二次加密后的密码: {encrypted_password}")
# # print(f"加密时间（time_num）: {time_num}")

# # # 使用
# # model_path = 'C:/Users/TCTrolong/Desktop/jiami/input/ajin.pth'  # 要加密的模型文件路径
# # output_path = 'C:/Users/TCTrolong/Desktop/jiami/output'  # 输出目录
# # password = encrypted_password  # 文件加密的密码

# # encrypt_model(model_path, password, key, output_extension='.niubi', output_dir=output_path)
