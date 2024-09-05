import os
import torch
import datetime
import io
import numpy as np
import pickle  # 新增导入
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, padding

# 获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE = os.path.join(CURRENT_DIR, 'jiemi_log.txt')

def write_to_log(content):
    """记录运行日志到log.txt文件"""
    with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
        log_file.write(content + '\n')

def calculate_encrypted_password(original_password, time_str):
    """
    根据输入的时间计算二次加密的密码。
    :param original_password: 用户输入的初始密码
    :param time_str: 加密时的时间字符串
    :return: 二次加密的密码
    """
    time_num = int(time_str)
    
    # 计算 time_num 的两倍
    time_numx2 = time_num * 2
    
    # 将初始密码每个字符的 ASCII 转换为字符串
    original_password_ascii = ''.join([str(ord(c)) for c in original_password])
    
    # 计算 original_password_ascii 的三倍
    original_password_asciix3 = int(original_password_ascii) * 3
    
    # 计算 add_num
    add_num = time_numx2 + original_password_asciix3
    
    # 将 add_num 从末尾开始每两位划分为一组，并倒序排列
    add_num_str = str(add_num)
    array_partitioning = [int(add_num_str[i:i+2]) for i in range(0, len(add_num_str), 2)][::-1]
    
    # 对应 ASCII 值转换为字符
    char_group = [chr(x) for x in array_partitioning]
    
    # 插入字符组
    second_encryption = ""
    remaining_numbers = str(add_num)
    index = 0

    for i in range(len(remaining_numbers)):
        second_encryption += remaining_numbers[i]
        # 根据位置决定插入字符
        if (i + 1) % (index + 1) == 0 and index < len(char_group):
            second_encryption += char_group[index]
            index += 1

    # 添加剩余字符（如果有）
    if index < len(char_group):
        second_encryption += ''.join(char_group[index:])

    return second_encryption

def decrypt_file_in_memory(ciphertext, password):
    """
    对文件内容进行 AES-256 解密，避免临时文件。
    :param ciphertext: 密文数据
    :param password: 用于解密的密码
    :return: 解密后的数据
    """
    salt = ciphertext[:16]
    iv = ciphertext[16:32]
    actual_ciphertext = ciphertext[32:]
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = kdf.derive(password.encode())

    # 初始化 AES-256 解密算法
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    # 解密数据
    padded_data = decryptor.update(actual_ciphertext) + decryptor.finalize()

    # 检查填充是否正确
    try:
        # 去除填充
        unpadder = padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded_data) + unpadder.finalize()
    except ValueError as e:
        print("去填充过程中发生错误：", e)
        print("解密数据长度:", len(padded_data))
        print("部分解密数据内容（前100字节）:", padded_data[:100])  # 输出部分解密数据以供调试
        write_to_log(f"解密失败，原因：{e}")
        raise e

    return plaintext

def complex_decrypt_tensor(tensor, key):
    """
    对模型的权重张量进行解密。
    :param tensor: 加密的张量
    :param key: 用于解密的key
    :return: 解密后的张量
    """
    tensor_np = tensor.cpu().numpy()
    perm = np.random.permutation(tensor_np.size)  # 保持随机种子相同
    decrypted_tensor = tensor_np.flatten()[np.argsort(perm)].reshape(tensor_np.shape)

    noise = np.random.normal(0, 0.1, tensor_np.shape)
    decrypted_tensor = (decrypted_tensor - noise * key)

    return torch.tensor(decrypted_tensor, dtype=tensor.dtype, device=tensor.device)

def decrypt_non_tensor(encrypted_data, password, key):
    """
    对非张量参数进行解密。
    :param encrypted_data: 加密的非张量数据
    :param password: 用于解密的密码
    :param key: 解密使用的key
    :return: 解密后的非张量数据
    """
    decrypted_data = decrypt_file_in_memory(encrypted_data, password)
    
    # 提取存储的 key 和实际数据
    key_bytes = decrypted_data[:8]
    extracted_key = int.from_bytes(key_bytes, byteorder='big', signed=True) / 1000.0  # 提取 key

    if extracted_key != key:
        # 不抛出异常，而是返回一个特殊值，表示解密失败
        return None  

    # 提取实际的数据
    serialized_data = decrypted_data[8:]
    value = pickle.loads(serialized_data)  # 使用pickle反序列化
    return value

def decrypt_model_weights(encrypted_state_dict, key, password):
    """
    对模型的每个权重矩阵和非张量参数进行复杂解密。
    :param encrypted_state_dict: 加密的模型状态字典
    :param key: 用于解密的key
    :param password: 用于解密的密码
    :return: 解密后的模型状态字典和解密信息
    """
    decrypted_state_dict = {}
    decrypted_params_info = []  # 用于记录解密的参数信息
    for name, param in encrypted_state_dict.items():
        if isinstance(param, torch.Tensor):
            decrypted_param = complex_decrypt_tensor(param, key)
            decrypted_state_dict[name] = decrypted_param
            decrypted_params_info.append(f"参数 {name} 已解密 - 类型: 张量")
        else:
            # 解密非张量类型参数，并确保key一致
            decrypted_param = decrypt_non_tensor(param, password, key)
            if decrypted_param is None:
                # 如果解密失败，返回 None
                print("秘钥错误，解密失败。")
                write_to_log("秘钥错误，解密失败。")
                return None, None
            decrypted_state_dict[name] = decrypted_param
            decrypted_params_info.append(f"参数 {name} 已解密 - 类型: {type(param)}")
    return decrypted_state_dict, decrypted_params_info

def decrypt_model(encrypted_file_path, original_password, time_str, key, output_path):
    """
    对加密的模型文件进行解密。
    :param encrypted_file_path: 加密的模型文件路径
    :param original_password: 用户输入的初始密码
    :param time_str: 加密时的时间（字符串格式）
    :param key: 用于解密的key
    :param output_path: 解密后的模型文件保存路径
    """
    # 读取加密后的模型文件
    with open(encrypted_file_path, 'rb') as f:
        encrypted_data = f.read()

    # 计算用户输入的密码进行加密验证
    calculated_encrypted_password = calculate_encrypted_password(original_password, time_str)
    print(f"计算出的二次加密密码: {calculated_encrypted_password}")
    write_to_log(f"计算出的二次加密密码: {calculated_encrypted_password}")

    # 尝试解密模型文件
    try:
        decrypted_data = decrypt_file_in_memory(encrypted_data, calculated_encrypted_password)
    except Exception as e:
        print("秘钥错误，解密失败。错误信息：", e)
        write_to_log(f"解密失败，错误信息：{e}")
        return

    # 加载解密后的模型字典
    buffer = io.BytesIO(decrypted_data)
    encrypted_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
    # 对模型权重进行复杂解密
    decrypted_state_dict, decrypted_params_info = decrypt_model_weights(encrypted_state_dict, key, calculated_encrypted_password)

    if decrypted_state_dict is None:
        # 解密失败，直接返回
        return

    # 保存解密后的模型文件
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")
        write_to_log(f"创建目录: {output_dir}")

    try:
        torch.save(decrypted_state_dict, output_path)
        print(f"模型解密成功并保存到 {output_path}")
        write_to_log(f"解密完成!\n原文件: {encrypted_file_path}\n解密后文件: {output_path}")
        write_to_log("\n".join(decrypted_params_info))
    except Exception as e:
        print(f"保存解密后的模型文件时发生错误: {e}")
        write_to_log(f"保存解密后的模型文件时发生错误: {e}")

# 输入和执行部分
encrypted_file_path = 'C:/Users/TCTrolong/Desktop/jiami/output/suannai40k.niubi'
original_password = input("请输入解密时的初始密码: ")
time_str = input("请输入加密时的时间（格式为 YYYYMMDDHHMM）：")
key = float(input("请输入矩阵加密的key（浮点数）："))
output_path = 'C:/Users/TCTrolong/Desktop/工作/项目/11rvc开发项目/模型加密&解密/自动/debug_output/suannai40k_jiemi.pth'

write_to_log(f"------------------------------------------------")  # 分隔符
write_to_log(f"开始解密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")

# 执行解密过程
decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)



# import os
# import torch
# import datetime
# import io
# import numpy as np
# import pickle  # 新增导入
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# from cryptography.hazmat.backends import default_backend
# from cryptography.hazmat.primitives import hashes, padding

# # 获取当前脚本所在目录
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# LOG_FILE = os.path.join(CURRENT_DIR, 'decrypt_log.txt')

# def write_to_log(content):
#     """记录运行日志到log.txt文件"""
#     with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
#         log_file.write(content + '\n')

# def calculate_encrypted_password(original_password, time_str):
#     """
#     根据输入的时间计算二次加密的密码。
#     :param original_password: 用户输入的初始密码
#     :param time_str: 加密时的时间字符串
#     :return: 二次加密的密码
#     """
#     time_num = int(time_str)
    
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

#     return second_encryption

# def decrypt_file_in_memory(ciphertext, password):
#     """
#     对文件内容进行 AES-256 解密，避免临时文件。
#     :param ciphertext: 密文数据
#     :param password: 用于解密的密码
#     :return: 解密后的数据
#     """
#     salt = ciphertext[:16]
#     iv = ciphertext[16:32]
#     actual_ciphertext = ciphertext[32:]
    
#     kdf = PBKDF2HMAC(
#         algorithm=hashes.SHA256(),
#         length=32,
#         salt=salt,
#         iterations=100000,
#         backend=default_backend()
#     )
#     key = kdf.derive(password.encode())

#     # 初始化 AES-256 解密算法
#     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
#     decryptor = cipher.decryptor()

#     # 解密数据
#     padded_data = decryptor.update(actual_ciphertext) + decryptor.finalize()

#     # 检查填充是否正确
#     try:
#         # 去除填充
#         unpadder = padding.PKCS7(128).unpadder()
#         plaintext = unpadder.update(padded_data) + unpadder.finalize()
#     except ValueError as e:
#         print("去填充过程中发生错误：", e)
#         print("解密数据长度:", len(padded_data))
#         print("部分解密数据内容（前100字节）:", padded_data[:100])  # 输出部分解密数据以供调试
#         write_to_log(f"解密失败，原因：{e}")
#         raise e

#     return plaintext

# def complex_decrypt_tensor(tensor, key):
#     """
#     对模型的权重张量进行解密。
#     :param tensor: 加密的张量
#     :param key: 用于解密的key
#     :return: 解密后的张量
#     """
#     tensor_np = tensor.cpu().numpy()
#     perm = np.random.permutation(tensor_np.size)  # 保持随机种子相同
#     decrypted_tensor = tensor_np.flatten()[np.argsort(perm)].reshape(tensor_np.shape)

#     noise = np.random.normal(0, 0.1, tensor_np.shape)
#     decrypted_tensor = (decrypted_tensor - noise * key)

#     return torch.tensor(decrypted_tensor, dtype=tensor.dtype, device=tensor.device)

# def decrypt_non_tensor(encrypted_data, password, key):
#     """
#     对非张量参数进行解密。
#     :param encrypted_data: 加密的非张量数据
#     :param password: 用于解密的密码
#     :param key: 解密使用的key
#     :return: 解密后的非张量数据
#     """
#     decrypted_data = decrypt_file_in_memory(encrypted_data, password)
    
#     # 提取存储的 key 和实际数据
#     key_bytes = decrypted_data[:8]
#     extracted_key = int.from_bytes(key_bytes, byteorder='big', signed=True) / 1000.0  # 提取 key

#     if extracted_key != key:
#         raise ValueError("解密失败，输入的key不正确！")  # 如果输入的key不正确，解密失败

#     # 提取实际的数据
#     serialized_data = decrypted_data[8:]
#     value = pickle.loads(serialized_data)  # 使用pickle反序列化
#     return value

# def decrypt_model_weights(encrypted_state_dict, key, password):
#     """
#     对模型的每个权重矩阵和非张量参数进行复杂解密。
#     :param encrypted_state_dict: 加密的模型状态字典
#     :param key: 用于解密的key
#     :param password: 用于解密的密码
#     :return: 解密后的模型状态字典和解密信息
#     """
#     decrypted_state_dict = {}
#     decrypted_params_info = []  # 用于记录解密的参数信息
#     for name, param in encrypted_state_dict.items():
#         if isinstance(param, torch.Tensor):
#             decrypted_param = complex_decrypt_tensor(param, key)
#             decrypted_state_dict[name] = decrypted_param
#             decrypted_params_info.append(f"参数 {name} 已解密 - 类型: 张量")
#         else:
#             # 解密非张量类型参数，并确保key一致
#             decrypted_state_dict[name] = decrypt_non_tensor(param, password, key)
#             decrypted_params_info.append(f"参数 {name} 已解密 - 类型: {type(param)}")
#     return decrypted_state_dict, decrypted_params_info

# def decrypt_model(encrypted_file_path, original_password, time_str, key, output_path):
#     """
#     对加密的模型文件进行解密。
#     :param encrypted_file_path: 加密的模型文件路径
#     :param original_password: 用户输入的初始密码
#     :param time_str: 加密时的时间（字符串格式）
#     :param key: 用于解密的key
#     :param output_path: 解密后的模型文件保存路径
#     """
#     # 读取加密后的模型文件
#     with open(encrypted_file_path, 'rb') as f:
#         encrypted_data = f.read()

#     # 计算用户输入的密码进行加密验证
#     calculated_encrypted_password = calculate_encrypted_password(original_password, time_str)
#     print(f"计算出的二次加密密码: {calculated_encrypted_password}")
#     write_to_log(f"计算出的二次加密密码: {calculated_encrypted_password}")

#     # 尝试解密模型文件
#     try:
#         decrypted_data = decrypt_file_in_memory(encrypted_data, calculated_encrypted_password)
#     except Exception as e:
#         print("秘钥错误，解密失败。错误信息：", e)
#         write_to_log(f"解密失败，错误信息：{e}")
#         return

#     # 加载解密后的模型字典
#     buffer = io.BytesIO(decrypted_data)
#     encrypted_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
#     # 对模型权重进行复杂解密
#     decrypted_state_dict, decrypted_params_info = decrypt_model_weights(encrypted_state_dict, key, calculated_encrypted_password)

#     # 保存解密后的模型文件
#     output_dir = os.path.dirname(output_path)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"创建目录: {output_dir}")
#         write_to_log(f"创建目录: {output_dir}")

#     try:
#         torch.save(decrypted_state_dict, output_path)
#         print(f"模型解密成功并保存到 {output_path}")
#         write_to_log(f"解密完成: 原文件: {encrypted_file_path}, 解密后文件: {output_path}")
#         write_to_log("\n".join(decrypted_params_info))
#     except Exception as e:
#         print(f"保存解密后的模型文件时发生错误: {e}")
#         write_to_log(f"保存解密后的模型文件时发生错误: {e}")

# # 输入和执行部分
# encrypted_file_path = 'C:/Users/TCTrolong/Desktop/jiami/output/suannai40k.niubi'
# original_password = input("请输入解密时的初始密码: ")
# time_str = input("请输入加密时的时间（格式为 YYYYMMDDHHMM）：")
# key = float(input("请输入矩阵加密的key（浮点数）："))
# output_path = 'C:/Users/TCTrolong/Desktop/工作/项目/11rvc开发项目/模型加密&解密/自动/debug_output/suannai40k_jiemi.pth'

# write_to_log(f"--------------------------")  # 分隔符
# write_to_log(f"开始解密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")

# # 执行解密过程
# decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)



# import os
# import torch
# import datetime
# import io
# import numpy as np
# import pickle  # 新增导入
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# from cryptography.hazmat.backends import default_backend
# from cryptography.hazmat.primitives import hashes, padding

# # 获取当前脚本所在目录
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# LOG_FILE = os.path.join(CURRENT_DIR, 'decrypt_log.txt')

# def write_to_log(content):
#     """记录运行日志到log.txt文件"""
#     with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
#         log_file.write(content + '\n')

# def calculate_encrypted_password(original_password, time_str):
#     # 根据输入的时间计算二次加密的密码
#     time_num = int(time_str)
    
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

#     return second_encryption

# def decrypt_file_in_memory(ciphertext, password):
#     """对文件内容进行 AES-256 解密，避免临时文件"""
#     salt = ciphertext[:16]
#     iv = ciphertext[16:32]
#     actual_ciphertext = ciphertext[32:]
    
#     kdf = PBKDF2HMAC(
#         algorithm=hashes.SHA256(),
#         length=32,
#         salt=salt,
#         iterations=100000,
#         backend=default_backend()
#     )
#     key = kdf.derive(password.encode())

#     # 初始化 AES-256 解密算法
#     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
#     decryptor = cipher.decryptor()

#     # 解密数据
#     padded_data = decryptor.update(actual_ciphertext) + decryptor.finalize()

#     # 检查填充是否正确
#     try:
#         # 去除填充
#         unpadder = padding.PKCS7(128).unpadder()
#         plaintext = unpadder.update(padded_data) + unpadder.finalize()
#     except ValueError as e:
#         print("去填充过程中发生错误：", e)
#         print("解密数据长度:", len(padded_data))
#         print("部分解密数据内容（前100字节）:", padded_data[:100])  # 输出部分解密数据以供调试
#         write_to_log(f"解密失败，原因：{e}")
#         raise e

#     return plaintext

# def complex_decrypt_tensor(tensor, key, seed):
#     """使用随机种子和key对张量进行解密"""
#     np.random.seed(seed)  # 使用与加密相同的随机种子
#     tensor_np = tensor.cpu().numpy()
#     perm = np.random.permutation(tensor_np.size)
#     decrypted_tensor = tensor_np.flatten()[np.argsort(perm)].reshape(tensor_np.shape)

#     noise = np.random.normal(0, 0.1, tensor_np.shape)
#     decrypted_tensor = (decrypted_tensor - noise * key)

#     return torch.tensor(decrypted_tensor, dtype=tensor.dtype, device=tensor.device)

# # def decrypt_non_tensor(encrypted_data, password):
# #     """对非张量参数进行解密"""
# #     decrypted_data = decrypt_file_in_memory(encrypted_data, password)
# #     value = pickle.loads(decrypted_data)  # 使用pickle反序列化
# #     return value

# # def decrypt_model_weights(encrypted_state_dict, key, password, seed):
# #     """对模型的每个权重矩阵和非张量参数进行复杂解密"""
# #     decrypted_state_dict = {}
# #     decrypted_params_info = []  # 用于记录解密的参数信息
# #     for name, param in encrypted_state_dict.items():
# #         if isinstance(param, torch.Tensor):
# #             decrypted_param = complex_decrypt_tensor(param, key, seed)
# #             decrypted_state_dict[name] = decrypted_param
# #             decrypted_params_info.append(f"参数 {name} 已解密 - 类型: 张量")
# #         else:
# #             # 解密非张量类型参数
# #             decrypted_state_dict[name] = decrypt_non_tensor(param, password)
# #             decrypted_params_info.append(f"参数 {name} 已解密 - 类型: {type(param)}")
# #     return decrypted_state_dict, decrypted_params_info


# # 解密程序修改部分
# def decrypt_non_tensor(encrypted_data, password, key):
#     """对非张量参数进行解密"""
#     decrypted_data_with_key = decrypt_file_in_memory(encrypted_data, password)
    
#     # 提取并验证 key
#     key_bytes = decrypted_data_with_key[:8]
#     expected_key_bytes = int(key * 1000).to_bytes(8, byteorder='big', signed=True)
#     if key_bytes != expected_key_bytes:
#         raise ValueError("Key 错误，无法解密非张量数据。")
    
#     # 提取实际的序列化数据
#     serialized_data = decrypted_data_with_key[8:]
#     value = pickle.loads(serialized_data)  # 使用pickle反序列化
#     return value

# def decrypt_model_weights(encrypted_state_dict, key, password, seed):
#     """对模型的每个权重矩阵和非张量参数进行复杂解密"""
#     decrypted_state_dict = {}
#     decrypted_params_info = []  # 用于记录解密的参数信息
#     for name, param in encrypted_state_dict.items():
#         if isinstance(param, torch.Tensor):
#             decrypted_param = complex_decrypt_tensor(param, key, seed)
#             decrypted_state_dict[name] = decrypted_param
#             decrypted_params_info.append(f"参数 {name} 已解密 - 类型: 张量")
#         else:
#             # 解密非张量类型参数，并验证 key
#             decrypted_state_dict[name] = decrypt_non_tensor(param, password, key)
#             decrypted_params_info.append(f"参数 {name} 已解密 - 类型: {type(param)}")
#     return decrypted_state_dict, decrypted_params_info


# def decrypt_model(encrypted_file_path, original_password, time_str, key, output_path):
#     # 读取加密后的模型文件
#     with open(encrypted_file_path, 'rb') as f:
#         encrypted_data = f.read()

#     # 计算用户输入的密码进行加密验证
#     calculated_encrypted_password = calculate_encrypted_password(original_password, time_str)
#     print(f"计算出的二次加密密码: {calculated_encrypted_password}")
#     write_to_log(f"计算出的二次加密密码: {calculated_encrypted_password}")

#     # 尝试解密模型文件
#     try:
#         decrypted_data = decrypt_file_in_memory(encrypted_data, calculated_encrypted_password)
#     except Exception as e:
#         print("秘钥错误，解密失败。错误信息：", e)
#         write_to_log(f"解密失败，错误信息：{e}")
#         return

#     # 读取随机种子
#     seed = int.from_bytes(decrypted_data[:4], 'big')
#     model_data = decrypted_data[4:]

#     # 加载解密后的模型字典
#     buffer = io.BytesIO(model_data)
#     encrypted_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
#     # 对模型权重进行复杂解密
#     decrypted_state_dict, decrypted_params_info = decrypt_model_weights(encrypted_state_dict, key, calculated_encrypted_password, seed)

#     # 保存解密后的模型文件
#     output_dir = os.path.dirname(output_path)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"创建目录: {output_dir}")
#         write_to_log(f"创建目录: {output_dir}")

#     try:
#         torch.save(decrypted_state_dict, output_path)
#         print(f"模型解密成功并保存到 {output_path}")
#         write_to_log(f"解密完成: 原文件: {encrypted_file_path}, 解密后文件: {output_path}")
#         write_to_log("\n".join(decrypted_params_info))
#     except Exception as e:
#         print(f"保存解密后的模型文件时发生错误: {e}")
#         write_to_log(f"保存解密后的模型文件时发生错误: {e}")

# # 输入和执行部分
# encrypted_file_path = 'C:/Users/TCTrolong/Desktop/jiami/output/suannai40k.niubi'
# original_password = input("请输入解密时的初始密码: ")
# time_str = input("请输入加密时的时间（格式为 YYYYMMDDHHMM）：")
# key = float(input("请输入矩阵加密的key（浮点数）："))
# output_path = 'C:/Users/TCTrolong/Desktop/工作/项目/11rvc开发项目/模型加密&解密/自动/debug_output/suannai40k_jiemi.pth'

# write_to_log(f"--------------------------")  # 分隔符
# write_to_log(f"开始解密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")

# decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)




# import os
# import torch
# import datetime
# import io
# import numpy as np
# import pickle  # 新增导入
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# from cryptography.hazmat.backends import default_backend
# from cryptography.hazmat.primitives import hashes, padding

# # 获取当前脚本所在目录
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# LOG_FILE = os.path.join(CURRENT_DIR, 'decrypt_log.txt')

# def write_to_log(content):
#     """记录运行日志到log.txt文件"""
#     with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
#         log_file.write(content + '\n')

# def calculate_encrypted_password(original_password, time_str):
#     # 根据输入的时间计算二次加密的密码
#     time_num = int(time_str)
    
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

#     return second_encryption

# def decrypt_file_in_memory(ciphertext, password):
#     """对文件内容进行 AES-256 解密，避免临时文件"""
#     salt = ciphertext[:16]
#     iv = ciphertext[16:32]
#     actual_ciphertext = ciphertext[32:]
    
#     kdf = PBKDF2HMAC(
#         algorithm=hashes.SHA256(),
#         length=32,
#         salt=salt,
#         iterations=100000,
#         backend=default_backend()
#     )
#     key = kdf.derive(password.encode())

#     # 初始化 AES-256 解密算法
#     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
#     decryptor = cipher.decryptor()

#     # 解密数据
#     padded_data = decryptor.update(actual_ciphertext) + decryptor.finalize()

#     # 检查填充是否正确
#     try:
#         # 去除填充
#         unpadder = padding.PKCS7(128).unpadder()
#         plaintext = unpadder.update(padded_data) + unpadder.finalize()
#     except ValueError as e:
#         print("去填充过程中发生错误：", e)
#         print("解密数据长度:", len(padded_data))
#         print("部分解密数据内容（前100字节）:", padded_data[:100])  # 输出部分解密数据以供调试
#         write_to_log(f"解密失败，原因：{e}")
#         raise e

#     return plaintext

# def complex_decrypt_tensor(tensor, key, seed):
#     """对模型的权重张量进行解密"""
#     tensor_np = tensor.cpu().numpy()
    
#     # 使用相同的种子来生成置换
#     np.random.seed(seed)
#     perm = np.random.permutation(tensor_np.size)
#     decrypted_tensor = tensor_np.flatten()[np.argsort(perm)].reshape(tensor_np.shape)

#     noise = np.random.normal(0, 0.1, tensor_np.shape)
#     decrypted_tensor = (decrypted_tensor - noise * key)

#     return torch.tensor(decrypted_tensor, dtype=tensor.dtype, device=tensor.device)

# def decrypt_non_tensor(encrypted_data, password):
#     """对非张量参数进行解密"""
#     decrypted_data = decrypt_file_in_memory(encrypted_data, password)
#     value = pickle.loads(decrypted_data)  # 使用pickle反序列化
#     return value

# def decrypt_model_weights(encrypted_state_dict, key, password, seed):
#     """对模型的每个权重矩阵和非张量参数进行复杂解密"""
#     decrypted_state_dict = {}
#     decrypted_params_info = []  # 用于记录解密的参数信息
#     for name, param in encrypted_state_dict.items():
#         if isinstance(param, torch.Tensor):
#             decrypted_param = complex_decrypt_tensor(param, key, seed)
#             decrypted_state_dict[name] = decrypted_param
#             decrypted_params_info.append(f"参数 {name} 已解密 - 类型: 张量")
#         else:
#             # 解密非张量类型参数
#             decrypted_state_dict[name] = decrypt_non_tensor(param, password)
#             decrypted_params_info.append(f"参数 {name} 已解密 - 类型: {type(param)}")
#     return decrypted_state_dict, decrypted_params_info

# def decrypt_model(encrypted_file_path, original_password, time_str, key, output_path):
#     # 读取加密后的模型文件
#     with open(encrypted_file_path, 'rb') as f:
#         encrypted_data = f.read()

#     # 计算用户输入的密码进行加密验证
#     calculated_encrypted_password = calculate_encrypted_password(original_password, time_str)
#     print(f"计算出的二次加密密码: {calculated_encrypted_password}")
#     write_to_log(f"计算出的二次加密密码: {calculated_encrypted_password}")

#     # 尝试解密模型文件
#     try:
#         decrypted_data = decrypt_file_in_memory(encrypted_data, calculated_encrypted_password)
#     except Exception as e:
#         print("秘钥错误，解密失败。错误信息：", e)
#         write_to_log(f"解密失败，错误信息：{e}")
#         return

#     # 加载解密后的模型字典
#     buffer = io.BytesIO(decrypted_data)
#     encrypted_dict = torch.load(buffer, map_location=torch.device('cpu'))
#     encrypted_state_dict = encrypted_dict['state_dict']
#     seed = encrypted_dict['seed']  # 读取存储的随机种子
    
#     # 对模型权重进行复杂解密
#     decrypted_state_dict, decrypted_params_info = decrypt_model_weights(encrypted_state_dict, key, calculated_encrypted_password, seed)

#     # 保存解密后的模型文件
#     output_dir = os.path.dirname(output_path)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"创建目录: {output_dir}")
#         write_to_log(f"创建目录: {output_dir}")

#     try:
#         torch.save(decrypted_state_dict, output_path)
#         print(f"模型解密成功并保存到 {output_path}")
#         write_to_log(f"解密完成: 原文件: {encrypted_file_path}, 解密后文件: {output_path}")
#         write_to_log("\n".join(decrypted_params_info))
#     except Exception as e:
#         print(f"保存解密后的模型文件时发生错误: {e}")
#         write_to_log(f"保存解密后的模型文件时发生错误: {e}")

# # 输入和执行部分
# encrypted_file_path = 'C:/Users/TCTrolong/Desktop/jiami/output/suannai40k.niubi'
# original_password = input("请输入解密时的初始密码: ")
# time_str = input("请输入加密时的时间（格式为 YYYYMMDDHHMM）：")
# key = float(input("请输入矩阵加密的key（浮点数）："))
# output_path = 'C:/Users/TCTrolong/Desktop/工作/项目/11rvc开发项目/模型加密&解密/自动/debug_output/suannai40k_jiemi.pth'

# write_to_log(f"--------------------------")  # 分隔符
# write_to_log(f"开始解密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")

# decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)



# import os
# import torch
# import datetime
# import io
# import numpy as np
# import pickle  # 新增导入
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# from cryptography.hazmat.backends import default_backend
# from cryptography.hazmat.primitives import hashes, padding

# # 获取当前脚本所在目录
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# LOG_FILE = os.path.join(CURRENT_DIR, 'decrypt_log.txt')

# def write_to_log(content):
#     """记录运行日志到log.txt文件"""
#     with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
#         log_file.write(content + '\n')

# def calculate_encrypted_password(original_password, time_str):
#     # 根据输入的时间计算二次加密的密码
#     time_num = int(time_str)
    
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

#     return second_encryption

# def decrypt_file_in_memory(ciphertext, password):
#     """对文件内容进行 AES-256 解密，避免临时文件"""
#     salt = ciphertext[:16]
#     iv = ciphertext[16:32]
#     actual_ciphertext = ciphertext[32:]
    
#     kdf = PBKDF2HMAC(
#         algorithm=hashes.SHA256(),
#         length=32,
#         salt=salt,
#         iterations=100000,
#         backend=default_backend()
#     )
#     key = kdf.derive(password.encode())

#     # 初始化 AES-256 解密算法
#     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
#     decryptor = cipher.decryptor()

#     # 解密数据
#     padded_data = decryptor.update(actual_ciphertext) + decryptor.finalize()

#     # 检查填充是否正确
#     try:
#         # 去除填充
#         unpadder = padding.PKCS7(128).unpadder()
#         plaintext = unpadder.update(padded_data) + unpadder.finalize()
#     except ValueError as e:
#         print("去填充过程中发生错误：", e)
#         print("解密数据长度:", len(padded_data))
#         print("部分解密数据内容（前100字节）:", padded_data[:100])  # 输出部分解密数据以供调试
#         write_to_log(f"解密失败，原因：{e}")
#         raise e

#     return plaintext

# def complex_decrypt_tensor(tensor, key):
#     """对模型的权重张量进行解密"""
#     tensor_np = tensor.cpu().numpy()
#     perm = np.random.permutation(tensor_np.size)
#     decrypted_tensor = tensor_np.flatten()[np.argsort(perm)].reshape(tensor_np.shape)

#     noise = np.random.normal(0, 0.1, tensor_np.shape)
#     decrypted_tensor = (decrypted_tensor - noise * key)

#     return torch.tensor(decrypted_tensor, dtype=tensor.dtype, device=tensor.device)

# def decrypt_non_tensor(encrypted_data, password):
#     """对非张量参数进行解密"""
#     decrypted_data = decrypt_file_in_memory(encrypted_data, password)
#     value = pickle.loads(decrypted_data)  # 使用pickle反序列化
#     return value

# def decrypt_model_weights(encrypted_state_dict, key, password):
#     """对模型的每个权重矩阵和非张量参数进行复杂解密"""
#     decrypted_state_dict = {}
#     decrypted_params_info = []  # 用于记录解密的参数信息
#     for name, param in encrypted_state_dict.items():
#         if isinstance(param, torch.Tensor):
#             decrypted_param = complex_decrypt_tensor(param, key)
#             decrypted_state_dict[name] = decrypted_param
#             decrypted_params_info.append(f"参数 {name} 已解密 - 类型: 张量")
#         else:
#             # 解密非张量类型参数
#             decrypted_state_dict[name] = decrypt_non_tensor(param, password)
#             decrypted_params_info.append(f"参数 {name} 已解密 - 类型: {type(param)}")
#     return decrypted_state_dict, decrypted_params_info

# def decrypt_model(encrypted_file_path, original_password, time_str, key, output_path):
#     # 读取加密后的模型文件
#     with open(encrypted_file_path, 'rb') as f:
#         encrypted_data = f.read()

#     # 计算用户输入的密码进行加密验证
#     calculated_encrypted_password = calculate_encrypted_password(original_password, time_str)
#     print(f"计算出的二次加密密码: {calculated_encrypted_password}")
#     write_to_log(f"计算出的二次加密密码: {calculated_encrypted_password}")

#     # 尝试解密模型文件
#     try:
#         decrypted_data = decrypt_file_in_memory(encrypted_data, calculated_encrypted_password)
#     except Exception as e:
#         print("秘钥错误，解密失败。错误信息：", e)
#         write_to_log(f"解密失败，错误信息：{e}")
#         return

#     # 加载解密后的模型字典
#     buffer = io.BytesIO(decrypted_data)
#     encrypted_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
#     # 对模型权重进行复杂解密
#     decrypted_state_dict, decrypted_params_info = decrypt_model_weights(encrypted_state_dict, key, calculated_encrypted_password)

#     # 保存解密后的模型文件
#     output_dir = os.path.dirname(output_path)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"创建目录: {output_dir}")
#         write_to_log(f"创建目录: {output_dir}")

#     try:
#         torch.save(decrypted_state_dict, output_path)
#         print(f"模型解密成功并保存到 {output_path}")
#         write_to_log(f"解密完成: 原文件: {encrypted_file_path}, 解密后文件: {output_path}")
#         write_to_log("\n".join(decrypted_params_info))
#     except Exception as e:
#         print(f"保存解密后的模型文件时发生错误: {e}")
#         write_to_log(f"保存解密后的模型文件时发生错误: {e}")

# # 输入和执行部分
# encrypted_file_path = 'C:/Users/TCTrolong/Desktop/jiami/output/suannai40k.niubi'
# original_password = input("请输入解密时的初始密码: ")
# time_str = input("请输入加密时的时间（格式为 YYYYMMDDHHMM）：")
# key = float(input("请输入矩阵加密的key（浮点数）："))
# output_path = 'C:/Users/TCTrolong/Desktop/工作/项目/11rvc开发项目/模型加密&解密/自动/debug_output/suannai40k_jiemi.pth'

# write_to_log(f"--------------------------")  # 分隔符
# write_to_log(f"开始解密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")


# decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)
