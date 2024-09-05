import os
import torch
import io
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, padding

def calculate_encrypted_password(original_password, time_str):
    time_num = int(time_str)
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

    return second_encryption

def decrypt_file_in_memory(ciphertext, password):
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

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    padded_data = decryptor.update(actual_ciphertext) + decryptor.finalize()

    try:
        unpadder = padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded_data) + unpadder.finalize()
    except ValueError as e:
        print("去填充过程中发生错误：", e)
        raise e

    return plaintext

def complex_decrypt_tensor(tensor, key):
    tensor_np = tensor.cpu().numpy()
    perm = np.random.permutation(tensor_np.size)
    decrypted_tensor = tensor_np.flatten()[np.argsort(perm)].reshape(tensor_np.shape)
    
    noise = np.random.normal(0, 0.1, tensor_np.shape)
    decrypted_tensor = (decrypted_tensor - noise * key)

    return torch.tensor(decrypted_tensor, dtype=tensor.dtype, device=tensor.device)

def decrypt_model_weights(encrypted_state_dict, key):
    decrypted_state_dict = {}
    for name, param in encrypted_state_dict.items():
        if isinstance(param, torch.Tensor):
            decrypted_param = complex_decrypt_tensor(param, key)
            decrypted_state_dict[name] = decrypted_param
        else:
            decrypted_state_dict[name] = param
    return decrypted_state_dict

def decrypt_model(encrypted_file_path, original_password, time_str, key, output_path):
    with open(encrypted_file_path, 'rb') as f:
        encrypted_data = f.read()

    calculated_encrypted_password = calculate_encrypted_password(original_password, time_str)
    print(f"计算出的二次加密密码: {calculated_encrypted_password}")

    try:
        decrypted_data = decrypt_file_in_memory(encrypted_data, calculated_encrypted_password)
    except Exception as e:
        print("秘钥错误，解密失败。错误信息：", e)
        return

    buffer = io.BytesIO(decrypted_data)
    encrypted_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
    decrypted_state_dict = decrypt_model_weights(encrypted_state_dict, key)

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    try:
        torch.save(decrypted_state_dict, output_path)
        print(f"模型解密成功并保存到 {output_path}")
    except Exception as e:
        print(f"保存解密后的模型文件时发生错误: {e}")

# 输入和执行部分
encrypted_file_path = 'C:/Users/TCTrolong/Desktop/jiemi/input/suannai40k.niubi'
original_password = input("请输入解密时的初始密码: ")
time_str = input("请输入加密时的时间（格式为 YYYYMMDDHHMM）：")
key = float(input("请输入矩阵加密的key（浮点数）："))
output_path = 'C:/Users/TCTrolong/Desktop/jiemi/output/suannai40k_jiemi_2.pth'

decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)


# import os
# import torch
# import io
# import numpy as np
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# from cryptography.hazmat.backends import default_backend
# from cryptography.hazmat.primitives import hashes, padding

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
#         raise e

#     return plaintext

# # def complex_decrypt_matrix(matrix, key):
# #     """对模型的权重矩阵进行解密"""
# #     if isinstance(matrix, torch.Tensor):
# #         matrix_np = matrix.cpu().numpy()
    
# #         # 反置换矩阵元素
# #         decrypted_matrix = np.random.permutation(matrix_np.flatten()).reshape(matrix_np.shape)
    
# #         # 反向应用非线性扰动
# #         noise = np.random.normal(0, 0.1, matrix_np.shape)
# #         decrypted_matrix = (decrypted_matrix - noise * key)

# #         # 返回解密后的矩阵
# #         return torch.tensor(decrypted_matrix, dtype=matrix.dtype, device=matrix.device)
# #     else:
# #         print(f"Skipping non-Tensor type during decryption: {type(matrix)}")
# #         return matrix

# def complex_decrypt_matrix(matrix, key):
#     """对模型的权重矩阵进行解密"""
#     if isinstance(matrix, torch.Tensor):
#         matrix_np = matrix.cpu().numpy()
    
#         # 反置换矩阵元素
#         perm = np.random.permutation(matrix_np.size)
#         decrypted_matrix = matrix_np.flatten()[np.argsort(perm)].reshape(matrix_np.shape)
    
#         noise = np.random.normal(0, 0.1, matrix_np.shape)
#         decrypted_matrix = (decrypted_matrix - noise * key)

#         return torch.tensor(decrypted_matrix, dtype=matrix.dtype, device=matrix.device)
#     else:
#         return matrix


# # def decrypt_model_weights(encrypted_state_dict, key):
# #     """对模型的每个权重矩阵进行复杂解密"""
# #     decrypted_state_dict = {}
# #     for name, param in encrypted_state_dict.items():
# #         if isinstance(param, torch.Tensor):
# #             decrypted_param = complex_decrypt_matrix(param, key)
# #             decrypted_state_dict[name] = decrypted_param
# #         elif isinstance(param, dict):  # 如果是嵌套的字典，递归处理
# #             decrypted_state_dict[name] = decrypt_model_weights(param, key)
# #         else:
# #             decrypted_state_dict[name] = param  # 未加密的参数直接保存
# #     return decrypted_state_dict

# def decrypt_model_weights(encrypted_state_dict, key):
#     """对模型的每个权重矩阵进行复杂解密"""
#     decrypted_state_dict = {}
#     for name, param in encrypted_state_dict.items():
#         if isinstance(param, torch.Tensor):
#             decrypted_param = complex_decrypt_matrix(param, key)
#             decrypted_state_dict[name] = decrypted_param
#         elif isinstance(param, dict):
#             decrypted_state_dict[name] = decrypt_model_weights(param, key)
#         else:
#             decrypted_state_dict[name] = param
#     return decrypted_state_dict

# # def decrypt_model(encrypted_file_path, original_password, time_str, key, output_path):
# #     # 读取加密后的模型文件
# #     with open(encrypted_file_path, 'rb') as f:
# #         encrypted_data = f.read()

# #     # 计算用户输入的密码进行加密验证
# #     calculated_encrypted_password = calculate_encrypted_password(original_password, time_str)
# #     print(f"计算出的二次加密密码: {calculated_encrypted_password}")  # 调试输出，确认密码计算正确

# #     # 尝试解密模型文件
# #     try:
# #         decrypted_data = decrypt_file_in_memory(encrypted_data, calculated_encrypted_password)
# #     except Exception as e:
# #         print("秘钥错误，解密失败。错误信息：", e)
# #         return

# #     # 加载解密后的模型字典
# #     buffer = io.BytesIO(decrypted_data)
# #     encrypted_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
# #     # 对模型权重进行复杂解密
# #     decrypted_state_dict = decrypt_model_weights(encrypted_state_dict, key)

# #     # 保存解密后的模型文件
# #     torch.save(decrypted_state_dict, output_path)
# #     print(f"模型解密成功并保存到 {output_path}")

# # 解密模型函数
# def decrypt_model(encrypted_file_path, original_password, time_str, key, output_path):
#     with open(encrypted_file_path, 'rb') as f:
#         encrypted_data = f.read()

#     calculated_encrypted_password = calculate_encrypted_password(original_password, time_str)
#     print(f"计算出的二次加密密码: {calculated_encrypted_password}")

#     try:
#         decrypted_data = decrypt_file_in_memory(encrypted_data, calculated_encrypted_password)
#     except Exception as e:
#         print("秘钥错误，解密失败。错误信息：", e)
#         return

#     buffer = io.BytesIO(decrypted_data)
#     encrypted_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
#     decrypted_state_dict = decrypt_model_weights(encrypted_state_dict, key)

#     output_dir = os.path.dirname(output_path)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"创建目录: {output_dir}")

#     try:
#         torch.save(decrypted_state_dict, output_path)
#         print(f"模型解密成功并保存到 {output_path}")
#     except Exception as e:
#         print(f"保存解密后的模型文件时发生错误: {e}")

# # 输入和执行部分
# encrypted_file_path = 'C:/Users/TCTrolong/Desktop/jiemi/input/ajin.niubi'
# original_password = input("请输入解密时的初始密码: ")
# time_str = input("请输入加密时的时间（格式为 YYYYMMDDHHMM）：")
# key = float(input("请输入矩阵加密的key（浮点数）："))
# output_path = 'C:/Users/TCTrolong/Desktop/jiemi/output/ajin_jiemi.pth'

# decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)



# # import os
# # import torch
# # import io
# # import numpy as np
# # from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# # from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# # from cryptography.hazmat.backends import default_backend
# # from cryptography.hazmat.primitives import hashes, padding

# # def calculate_encrypted_password(original_password, time_str):
# #     # 根据输入的时间计算二次加密的密码
# #     time_num = int(time_str)
    
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

# #     return second_encryption

# # def decrypt_file_in_memory(ciphertext, password):
# #     """对文件内容进行 AES-256 解密，避免临时文件"""
# #     salt = ciphertext[:16]
# #     iv = ciphertext[16:32]
# #     actual_ciphertext = ciphertext[32:]
    
# #     kdf = PBKDF2HMAC(
# #         algorithm=hashes.SHA256(),
# #         length=32,
# #         salt=salt,
# #         iterations=100000,
# #         backend=default_backend()
# #     )
# #     key = kdf.derive(password.encode())

# #     # 初始化 AES-256 解密算法
# #     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
# #     decryptor = cipher.decryptor()

# #     # 解密数据
# #     padded_data = decryptor.update(actual_ciphertext) + decryptor.finalize()

# #     # 检查填充是否正确
# #     try:
# #         # 去除填充
# #         unpadder = padding.PKCS7(128).unpadder()
# #         plaintext = unpadder.update(padded_data) + unpadder.finalize()
# #     except ValueError as e:
# #         print("去填充过程中发生错误：", e)
# #         print("解密数据长度:", len(padded_data))
# #         print("部分解密数据内容（前100字节）:", padded_data[:100])  # 输出部分解密数据以供调试
# #         raise e

# #     return plaintext

# # # def complex_decrypt_matrix(matrix, key):
# # #     """对模型的权重矩阵进行解密"""
# # #     matrix_np = matrix.cpu().numpy()
    
# # #     # 反置换矩阵元素
# # #     decrypted_matrix = np.random.permutation(matrix_np.flatten()).reshape(matrix_np.shape)
    
# # #     # 反向应用非线性扰动
# # #     noise = np.random.normal(0, 0.1, matrix_np.shape)
# # #     decrypted_matrix = (decrypted_matrix - noise * key)

# # #     # 返回解密后的矩阵
# # #     return torch.tensor(decrypted_matrix, dtype=matrix.dtype, device=matrix.device)

# # def complex_decrypt_matrix(matrix, key):
# #     """对模型的权重矩阵进行解密"""
# #     if isinstance(matrix, torch.Tensor):
# #         matrix_np = matrix.cpu().numpy()
    
# #         # 反置换矩阵元素
# #         decrypted_matrix = np.random.permutation(matrix_np.flatten()).reshape(matrix_np.shape)
    
# #         # 反向应用非线性扰动
# #         noise = np.random.normal(0, 0.1, matrix_np.shape)
# #         decrypted_matrix = (decrypted_matrix - noise * key)

# #         # 返回解密后的矩阵
# #         return torch.tensor(decrypted_matrix, dtype=matrix.dtype, device=matrix.device)
# #     else:
# #         print(f"Skipping non-Tensor type during decryption: {type(matrix)}")
# #         return matrix


# # def decrypt_model(encrypted_file_path, original_password, time_str, key, output_path):
# #     # 读取加密后的模型文件
# #     with open(encrypted_file_path, 'rb') as f:
# #         encrypted_data = f.read()

# #     # 计算用户输入的密码进行加密验证
# #     calculated_encrypted_password = calculate_encrypted_password(original_password, time_str)
# #     print(f"计算出的二次加密密码: {calculated_encrypted_password}")  # 调试输出，确认密码计算正确

# #     # 尝试解密模型文件
# #     try:
# #         decrypted_data = decrypt_file_in_memory(encrypted_data, calculated_encrypted_password)
# #     except Exception as e:
# #         print("秘钥错误，解密失败。错误信息：", e)
# #         return

# #     # 加载解密后的模型字典
# #     buffer = io.BytesIO(decrypted_data)
# #     encrypted_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
# #     # 对模型权重进行复杂解密
# #     decrypted_state_dict = {}
# #     for name, param in encrypted_state_dict.items():
# #         if isinstance(param, torch.Tensor):
# #             decrypted_param = complex_decrypt_matrix(param, key)
# #             decrypted_state_dict[name] = decrypted_param
# #         else:
# #             decrypted_state_dict[name] = param  # 未加密的参数直接保存

# #     # 确保输出路径是一个有效的文件路径
# #     if not os.path.isdir(output_path):
# #         try:
# #             os.makedirs(output_path)  # 尝试创建目录
# #             print(f"创建目录: {output_path}")
# #         except Exception as e:
# #             print(f"创建目录失败: {e}")
# #             return
    
# #     # 构建解密后的模型文件路径
# #     model_file_path = os.path.join(output_path, 'decrypted_model.pth')

# #     # 保存解密后的模型文件
# #     try:
# #         torch.save(decrypted_state_dict, model_file_path)
# #         print(f"模型解密成功并保存到 {model_file_path}")
# #     except Exception as e:
# #         print(f"保存解密后的模型文件时发生错误: {e}")

# # # 输入和执行部分
# # encrypted_file_path = 'C:/Users/TCTrolong/Desktop/jiemi/input/ajin.niubi'
# # original_password = input("请输入解密时的初始密码: ")
# # time_str = input("请输入加密时的时间（格式为 YYYYMMDDHHMM）：")
# # key = float(input("请输入矩阵加密的key（浮点数）："))
# # output_path = 'C:/Users/TCTrolong/Desktop/jiemi/output'

# # decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)


# # # import os
# # # import torch
# # # import io
# # # import numpy as np
# # # from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# # # from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# # # from cryptography.hazmat.backends import default_backend
# # # from cryptography.hazmat.primitives import hashes, padding

# # # def calculate_encrypted_password(original_password, time_str):
# # #     # 根据输入的时间计算二次加密的密码
# # #     time_num = int(time_str)
    
# # #     # 计算 time_num 的两倍
# # #     time_numx2 = time_num * 2
    
# # #     # 将初始密码每个字符的 ASCII 转换为字符串
# # #     original_password_ascii = ''.join([str(ord(c)) for c in original_password])
    
# # #     # 计算 original_password_ascii 的三倍
# # #     original_password_asciix3 = int(original_password_ascii) * 3
    
# # #     # 计算 add_num
# # #     add_num = time_numx2 + original_password_asciix3
    
# # #     # 将 add_num 从末尾开始每两位划分为一组，并倒序排列
# # #     add_num_str = str(add_num)
# # #     array_partitioning = [int(add_num_str[i:i+2]) for i in range(0, len(add_num_str), 2)][::-1]
    
# # #     # 对应 ASCII 值转换为字符
# # #     char_group = [chr(x) for x in array_partitioning]
    
# # #     # 插入字符组
# # #     second_encryption = ""
# # #     remaining_numbers = str(add_num)
# # #     index = 0

# # #     for i in range(len(remaining_numbers)):
# # #         second_encryption += remaining_numbers[i]
# # #         # 根据位置决定插入字符
# # #         if (i + 1) % (index + 1) == 0 and index < len(char_group):
# # #             second_encryption += char_group[index]
# # #             index += 1

# # #     # 添加剩余字符（如果有）
# # #     if index < len(char_group):
# # #         second_encryption += ''.join(char_group[index:])

# # #     return second_encryption

# # # def decrypt_file_in_memory(ciphertext, password):
# # #     """对文件内容进行 AES-256 解密，避免临时文件"""
# # #     salt = ciphertext[:16]
# # #     iv = ciphertext[16:32]
# # #     actual_ciphertext = ciphertext[32:]
    
# # #     kdf = PBKDF2HMAC(
# # #         algorithm=hashes.SHA256(),
# # #         length=32,
# # #         salt=salt,
# # #         iterations=100000,
# # #         backend=default_backend()
# # #     )
# # #     key = kdf.derive(password.encode())

# # #     # 初始化 AES-256 解密算法
# # #     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
# # #     decryptor = cipher.decryptor()

# # #     # 解密数据
# # #     padded_data = decryptor.update(actual_ciphertext) + decryptor.finalize()

# # #     # 检查填充是否正确
# # #     try:
# # #         # 去除填充
# # #         unpadder = padding.PKCS7(128).unpadder()
# # #         plaintext = unpadder.update(padded_data) + unpadder.finalize()
# # #     except ValueError as e:
# # #         print("去填充过程中发生错误：", e)
# # #         print("解密数据长度:", len(padded_data))
# # #         print("部分解密数据内容（前100字节）:", padded_data[:100])  # 输出部分解密数据以供调试
# # #         raise e

# # #     return plaintext

# # # def complex_decrypt_matrix(matrix, key):
# # #     """对模型的权重矩阵进行解密"""
# # #     matrix_np = matrix.cpu().numpy()
    
# # #     # 反置换矩阵元素
# # #     decrypted_matrix = np.random.permutation(matrix_np.flatten()).reshape(matrix_np.shape)
    
# # #     # 反向应用非线性扰动
# # #     noise = np.random.normal(0, 0.1, matrix_np.shape)
# # #     decrypted_matrix = (decrypted_matrix - noise * key)

# # #     # 返回解密后的矩阵
# # #     return torch.tensor(decrypted_matrix, dtype=matrix.dtype, device=matrix.device)

# # # def decrypt_model(encrypted_file_path, original_password, time_str, key, output_path):
# # #     # 读取加密后的模型文件
# # #     with open(encrypted_file_path, 'rb') as f:
# # #         encrypted_data = f.read()

# # #     # 计算用户输入的密码进行加密验证
# # #     calculated_encrypted_password = calculate_encrypted_password(original_password, time_str)
# # #     print(f"计算出的二次加密密码: {calculated_encrypted_password}")  # 调试输出，确认密码计算正确

# # #     # 尝试解密模型文件
# # #     try:
# # #         decrypted_data = decrypt_file_in_memory(encrypted_data, calculated_encrypted_password)
# # #     except Exception as e:
# # #         print("秘钥错误，解密失败。错误信息：", e)
# # #         return

# # #     # 加载解密后的模型字典
# # #     buffer = io.BytesIO(decrypted_data)
# # #     encrypted_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
# # #     # 对模型权重进行复杂解密
# # #     decrypted_state_dict = {}
# # #     for name, param in encrypted_state_dict.items():
# # #         if isinstance(param, torch.Tensor):
# # #             decrypted_param = complex_decrypt_matrix(param, key)
# # #             decrypted_state_dict[name] = decrypted_param
# # #         else:
# # #             decrypted_state_dict[name] = param  # 未加密的参数直接保存

# # #     # 保存解密后的模型文件
# # #     torch.save(decrypted_state_dict, output_path)
# # #     print(f"模型解密成功并保存到 {output_path}")

# # # # # 输入和执行部分
# # # # encrypted_file_path = input("请输入加密的模型文件路径: ")
# # # # original_password = input("请输入解密时的初始密码: ")
# # # # time_str = input("请输入加密时的时间（格式为 YYYYMMDDHHMM）：")
# # # # key = float(input("请输入矩阵加密的key（浮点数）："))
# # # # output_path = input("请输入解密后模型的保存路径: ")

# # # # 输入和执行部分
# # # encrypted_file_path = 'C:/Users/TCTrolong/Desktop/工作/项目/11rvc开发项目/模型加密&解密/加密测试文件/Output/suannai40k.niubi'
# # # #input("请输入加密的模型文件路径: ")
# # # original_password = input("请输入解密时的初始密码: ")
# # # time_str = input("请输入加密时的时间（格式为 YYYYMMDDHHMM）：")
# # # key = float(input("请输入矩阵加密的key（浮点数）："))
# # # output_path = 'C:/Users/TCTrolong/Desktop/工作/项目/11rvc开发项目/模型加密&解密/加密测试文件/解密输出测试'
# # # #input("请输入解密后模型的保存路径: ")

# # # decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)






# # # import os
# # # import torch
# # # from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# # # from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# # # from cryptography.hazmat.backends import default_backend
# # # from cryptography.hazmat.primitives import hashes, padding
# # # import io

# # # def decrypt_model(encrypted_file_path, password, debug_output_dir=None):
# # #     # 读取加密文件内容
# # #     with open(encrypted_file_path, 'rb') as f:
# # #         salt = f.read(16)  # 读取盐值
# # #         iv = f.read(16)  # 读取初始化向量
# # #         ciphertext = f.read()  # 读取密文

# # #     # 使用同样的 KDF 生成密钥
# # #     kdf = PBKDF2HMAC(
# # #         algorithm=hashes.SHA256(),
# # #         length=32,
# # #         salt=salt,
# # #         iterations=100000,
# # #         backend=default_backend()
# # #     )
# # #     key = kdf.derive(password.encode())

# # #     # 初始化解密算法
# # #     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
# # #     decryptor = cipher.decryptor()

# # #     # 解密数据
# # #     padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

# # #     # 去除填充
# # #     unpadder = padding.PKCS7(128).unpadder()
# # #     plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

# # #     # 调试功能：将解密后的文件保存到指定目录
# # #     if debug_output_dir:
# # #         debug_output_path = os.path.join(debug_output_dir, os.path.basename(encrypted_file_path) + '.decrypted.pth')
# # #         with open(debug_output_path, 'wb') as debug_file:
# # #             debug_file.write(plaintext)
# # #         print(f"Decrypted model saved to {debug_output_path} for debugging purposes.")

# # #     # 使用 torch 加载模型
# # #     model = torch.load(io.BytesIO(plaintext))

# # #     print(f"Model decrypted and loaded from {encrypted_file_path}")
# # #     return model

# # # # 示例使用
# # # # 请替换为你的加密文件路径和调试输出目录
# # # encrypted_model_path = 'C:/Users/TCTrolong/Desktop/工作/项目/11rvc开发项目/模型加密&解密/加密测试文件/Output/ajin.niubi'  # 加密后的文件路径
# # # debug_output_directory = 'C:/Users/TCTrolong/Desktop/工作/项目/11rvc开发项目/模型加密&解密/加密测试文件/解密输出测试'  # 调试输出目录
# # # model = decrypt_model(encrypted_model_path, '123456', debug_output_dir=debug_output_directory)   #文本参数为加密密码

# # # # 现在你可以使用 model 进行推理或进一步操作
# # # # 例如 model.eval() 等操作


# # # import os
# # # import torch
# # # import io
# # # import datetime
# # # import numpy as np
# # # from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# # # from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# # # from cryptography.hazmat.backends import default_backend
# # # from cryptography.hazmat.primitives import hashes, padding

# # # def calculate_encrypted_password(original_password, time_str):
# # #     # 根据加密时的逻辑重新生成二次加密后的密码
# # #     # 解析输入的时间
# # #     time_num = int(time_str)
    
# # #     # 计算 time_num 的两倍
# # #     time_numx2 = time_num * 2
    
# # #     # 将初始密码每个字符的 ASCII 转换为字符串
# # #     original_password_ascii = ''.join([str(ord(c)) for c in original_password])
    
# # #     # 计算 original_password_ascii 的三倍
# # #     original_password_asciix3 = int(original_password_ascii) * 3
    
# # #     # 计算 add_num
# # #     add_num = time_numx2 + original_password_asciix3
    
# # #     # 将 add_num 从末尾开始每两位划分为一组，并倒序排列
# # #     add_num_str = str(add_num)
# # #     array_partitioning = [int(add_num_str[i:i+2]) for i in range(0, len(add_num_str), 2)][::-1]
    
# # #     # 对应 ASCII 值转换为字符
# # #     char_group = [chr(x) for x in array_partitioning]
    
# # #     # 插入字符组
# # #     second_encryption = ""
# # #     remaining_numbers = str(add_num)
# # #     index = 0

# # #     for i in range(len(remaining_numbers)):
# # #         second_encryption += remaining_numbers[i]
# # #         # 根据位置决定插入字符
# # #         if (i + 1) % (index + 1) == 0 and index < len(char_group):
# # #             second_encryption += char_group[index]
# # #             index += 1

# # #     # 添加剩余字符（如果有）
# # #     if index < len(char_group):
# # #         second_encryption += ''.join(char_group[index:])

# # #     return second_encryption

# # # def decrypt_file_in_memory(ciphertext, password):
# # #     """对文件内容进行 AES-256 解密，避免临时文件"""
# # #     salt = ciphertext[:16]
# # #     iv = ciphertext[16:32]
# # #     actual_ciphertext = ciphertext[32:]
    
# # #     kdf = PBKDF2HMAC(
# # #         algorithm=hashes.SHA256(),
# # #         length=32,
# # #         salt=salt,
# # #         iterations=100000,
# # #         backend=default_backend()
# # #     )
# # #     key = kdf.derive(password.encode())

# # #     # 初始化 AES-256 解密算法
# # #     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
# # #     decryptor = cipher.decryptor()

# # #     # 解密数据
# # #     padded_data = decryptor.update(actual_ciphertext) + decryptor.finalize()

# # #     # 去除填充
# # #     unpadder = padding.PKCS7(128).unpadder()
# # #     plaintext = unpadder.update(padded_data) + unpadder.finalize()

# # #     return plaintext

# # # def complex_decrypt_matrix(matrix, key):
# # #     """对模型的权重矩阵进行解密"""
# # #     matrix_np = matrix.cpu().numpy()
    
# # #     # 反置换矩阵元素
# # #     decrypted_matrix = np.random.permutation(matrix_np.flatten()).reshape(matrix_np.shape)
    
# # #     # 反向应用非线性扰动
# # #     noise = np.random.normal(0, 0.1, matrix_np.shape)
# # #     decrypted_matrix = (decrypted_matrix - noise * key)

# # #     # 返回解密后的矩阵
# # #     return torch.tensor(decrypted_matrix, dtype=matrix.dtype, device=matrix.device)

# # # def decrypt_model(encrypted_file_path, original_password, time_str, key, output_path):
# # #     # 读取加密后的模型文件
# # #     with open(encrypted_file_path, 'rb') as f:
# # #         encrypted_data = f.read()

# # #     # 计算用户输入的密码进行加密验证
# # #     calculated_encrypted_password = calculate_encrypted_password(original_password, time_str)

# # #     # 检查加密后的密码是否正确
# # #     encrypted_password_from_file = encrypted_data[:256].decode('utf-8')  # 假设加密后的密码存储在文件前256字节内
# # #     if calculated_encrypted_password != encrypted_password_from_file:
# # #         print("密码错误，解密失败")
# # #         return

# # #     # 解密模型文件
# # #     try:
# # #         decrypted_data = decrypt_file_in_memory(encrypted_data[256:], calculated_encrypted_password)
# # #     except Exception as e:
# # #         print("密码错误，解密失败")
# # #         return

# # #     # 加载解密后的模型字典
# # #     buffer = io.BytesIO(decrypted_data)
# # #     encrypted_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
# # #     # 对模型权重进行复杂解密
# # #     decrypted_state_dict = {}
# # #     for name, param in encrypted_state_dict.items():
# # #         if param.requires_grad:
# # #             decrypted_param = complex_decrypt_matrix(param, key)
# # #             decrypted_state_dict[name] = decrypted_param
# # #         else:
# # #             decrypted_state_dict[name] = param  # 未加密的参数直接保存

# # #     # 保存解密后的模型文件
# # #     torch.save(decrypted_state_dict, output_path)
# # #     print(f"模型解密成功并保存到 {output_path}")

# # # # 输入和执行部分
# # # encrypted_file_path = input("请输入加密的模型文件路径: ")
# # # original_password = input("请输入解密时的初始密码: ")
# # # time_str = input("请输入加密时的时间（格式为 YYYYMMDDHHMM）：")
# # # key = float(input("请输入矩阵加密的key（浮点数）："))
# # # output_path = input("请输入解密后模型的保存路径: ")

# # # decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)


# # # import os
# # # import torch
# # # import io
# # # import datetime
# # # import numpy as np
# # # from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# # # from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# # # from cryptography.hazmat.backends import default_backend
# # # from cryptography.hazmat.primitives import hashes, padding

# # # def calculate_encrypted_password(original_password, time_str):
# # #     # 根据输入的时间计算二次加密的密码
# # #     time_num = int(time_str)
    
# # #     # 计算 time_num 的两倍
# # #     time_numx2 = time_num * 2
    
# # #     # 将初始密码每个字符的 ASCII 转换为字符串
# # #     original_password_ascii = ''.join([str(ord(c)) for c in original_password])
    
# # #     # 计算 original_password_ascii 的三倍
# # #     original_password_asciix3 = int(original_password_ascii) * 3
    
# # #     # 计算 add_num
# # #     add_num = time_numx2 + original_password_asciix3
    
# # #     # 将 add_num 从末尾开始每两位划分为一组，并倒序排列
# # #     add_num_str = str(add_num)
# # #     array_partitioning = [int(add_num_str[i:i+2]) for i in range(0, len(add_num_str), 2)][::-1]
    
# # #     # 对应 ASCII 值转换为字符
# # #     char_group = [chr(x) for x in array_partitioning]
    
# # #     # 插入字符组
# # #     second_encryption = ""
# # #     remaining_numbers = str(add_num)
# # #     index = 0

# # #     for i in range(len(remaining_numbers)):
# # #         second_encryption += remaining_numbers[i]
# # #         # 根据位置决定插入字符
# # #         if (i + 1) % (index + 1) == 0 and index < len(char_group):
# # #             second_encryption += char_group[index]
# # #             index += 1

# # #     # 添加剩余字符（如果有）
# # #     if index < len(char_group):
# # #         second_encryption += ''.join(char_group[index:])

# # #     return second_encryption

# # # def decrypt_file_in_memory(ciphertext, password):
# # #     """对文件内容进行 AES-256 解密，避免临时文件"""
# # #     salt = ciphertext[:16]
# # #     iv = ciphertext[16:32]
# # #     actual_ciphertext = ciphertext[32:]
    
# # #     kdf = PBKDF2HMAC(
# # #         algorithm=hashes.SHA256(),
# # #         length=32,
# # #         salt=salt,
# # #         iterations=100000,
# # #         backend=default_backend()
# # #     )
# # #     key = kdf.derive(password.encode())

# # #     # 初始化 AES-256 解密算法
# # #     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
# # #     decryptor = cipher.decryptor()

# # #     # 解密数据
# # #     padded_data = decryptor.update(actual_ciphertext) + decryptor.finalize()

# # #     # 去除填充
# # #     unpadder = padding.PKCS7(128).unpadder()
# # #     plaintext = unpadder.update(padded_data) + unpadder.finalize()

# # #     return plaintext

# # # def complex_decrypt_matrix(matrix, key):
# # #     """对模型的权重矩阵进行解密"""
# # #     matrix_np = matrix.cpu().numpy()
    
# # #     # 反置换矩阵元素
# # #     decrypted_matrix = np.random.permutation(matrix_np.flatten()).reshape(matrix_np.shape)
    
# # #     # 反向应用非线性扰动
# # #     noise = np.random.normal(0, 0.1, matrix_np.shape)
# # #     decrypted_matrix = (decrypted_matrix - noise * key)

# # #     # 返回解密后的矩阵
# # #     return torch.tensor(decrypted_matrix, dtype=matrix.dtype, device=matrix.device)

# # # def decrypt_model(encrypted_file_path, original_password, time_str, key, output_path):
# # #     # 读取加密后的模型文件
# # #     with open(encrypted_file_path, 'rb') as f:
# # #         encrypted_data = f.read()

# # #     # 计算用户输入的密码进行加密验证
# # #     calculated_encrypted_password = calculate_encrypted_password(original_password, time_str)

# # #     # 解密模型文件
# # #     try:
# # #         decrypted_data = decrypt_file_in_memory(encrypted_data, calculated_encrypted_password)
# # #     except Exception as e:
# # #         print("秘钥错误，解密失败")
# # #         return

# # #     # 加载解密后的模型字典
# # #     buffer = io.BytesIO(decrypted_data)
# # #     encrypted_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
# # #     # 对模型权重进行复杂解密
# # #     decrypted_state_dict = {}
# # #     for name, param in encrypted_state_dict.items():
# # #         if isinstance(param, torch.Tensor):
# # #             decrypted_param = complex_decrypt_matrix(param, key)
# # #             decrypted_state_dict[name] = decrypted_param
# # #         else:
# # #             decrypted_state_dict[name] = param  # 未加密的参数直接保存

# # #     # 保存解密后的模型文件
# # #     torch.save(decrypted_state_dict, output_path)
# # #     print(f"模型解密成功并保存到 {output_path}")

# # # # 输入和执行部分
# # # encrypted_file_path = 'C:/Users/TCTrolong/Desktop/工作/项目/11rvc开发项目/模型加密&解密/加密测试文件/Output/suannai40k.niubi'
# # # #input("请输入加密的模型文件路径: ")
# # # original_password = input("请输入解密时的初始密码: ")
# # # time_str = input("请输入加密时的时间（格式为 YYYYMMDDHHMM）：")
# # # key = float(input("请输入矩阵加密的key（浮点数）："))
# # # output_path = 'C:/Users/TCTrolong/Desktop/工作/项目/11rvc开发项目/模型加密&解密/加密测试文件/解密输出测试'
# # # #input("请输入解密后模型的保存路径: ")

# # # decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)


# # # import os
# # # import torch
# # # import io
# # # import datetime
# # # import numpy as np
# # # from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# # # from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# # # from cryptography.hazmat.backends import default_backend
# # # from cryptography.hazmat.primitives import hashes, padding

# # # def calculate_encrypted_password(original_password, time_str):
# # #     # 根据输入的时间计算二次加密的密码
# # #     time_num = int(time_str)
    
# # #     # 计算 time_num 的两倍
# # #     time_numx2 = time_num * 2
    
# # #     # 将初始密码每个字符的 ASCII 转换为字符串
# # #     original_password_ascii = ''.join([str(ord(c)) for c in original_password])
    
# # #     # 计算 original_password_ascii 的三倍
# # #     original_password_asciix3 = int(original_password_ascii) * 3
    
# # #     # 计算 add_num
# # #     add_num = time_numx2 + original_password_asciix3
    
# # #     # 将 add_num 从末尾开始每两位划分为一组，并倒序排列
# # #     add_num_str = str(add_num)
# # #     array_partitioning = [int(add_num_str[i:i+2]) for i in range(0, len(add_num_str), 2)][::-1]
    
# # #     # 对应 ASCII 值转换为字符
# # #     char_group = [chr(x) for x in array_partitioning]
    
# # #     # 插入字符组
# # #     second_encryption = ""
# # #     remaining_numbers = str(add_num)
# # #     index = 0

# # #     for i in range(len(remaining_numbers)):
# # #         second_encryption += remaining_numbers[i]
# # #         # 根据位置决定插入字符
# # #         if (i + 1) % (index + 1) == 0 and index < len(char_group):
# # #             second_encryption += char_group[index]
# # #             index += 1

# # #     # 添加剩余字符（如果有）
# # #     if index < len(char_group):
# # #         second_encryption += ''.join(char_group[index:])

# # #     return second_encryption

# # # def decrypt_file_in_memory(ciphertext, password):
# # #     """对文件内容进行 AES-256 解密，避免临时文件"""
# # #     salt = ciphertext[:16]
# # #     iv = ciphertext[16:32]
# # #     actual_ciphertext = ciphertext[32:]
    
# # #     kdf = PBKDF2HMAC(
# # #         algorithm=hashes.SHA256(),
# # #         length=32,
# # #         salt=salt,
# # #         iterations=100000,
# # #         backend=default_backend()
# # #     )
# # #     key = kdf.derive(password.encode())

# # #     # 初始化 AES-256 解密算法
# # #     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
# # #     decryptor = cipher.decryptor()

# # #     # 解密数据
# # #     padded_data = decryptor.update(actual_ciphertext) + decryptor.finalize()

# # #     # 去除填充
# # #     unpadder = padding.PKCS7(128).unpadder()
# # #     plaintext = unpadder.update(padded_data) + unpadder.finalize()

# # #     return plaintext

# # # def complex_decrypt_matrix(matrix, key):
# # #     """对模型的权重矩阵进行解密"""
# # #     matrix_np = matrix.cpu().numpy()
    
# # #     # 反置换矩阵元素
# # #     decrypted_matrix = np.random.permutation(matrix_np.flatten()).reshape(matrix_np.shape)
    
# # #     # 反向应用非线性扰动
# # #     noise = np.random.normal(0, 0.1, matrix_np.shape)
# # #     decrypted_matrix = (decrypted_matrix - noise * key)

# # #     # 返回解密后的矩阵
# # #     return torch.tensor(decrypted_matrix, dtype=matrix.dtype, device=matrix.device)

# # # def decrypt_model(encrypted_file_path, original_password, time_str, key, output_path):
# # #     # 读取加密后的模型文件
# # #     with open(encrypted_file_path, 'rb') as f:
# # #         encrypted_data = f.read()

# # #     # 计算用户输入的密码进行加密验证
# # #     calculated_encrypted_password = calculate_encrypted_password(original_password, time_str)
# # #     print(f"计算出的二次加密密码: {calculated_encrypted_password}")  # 调试输出，确认密码计算正确

# # #     # 尝试解密模型文件
# # #     try:
# # #         decrypted_data = decrypt_file_in_memory(encrypted_data, calculated_encrypted_password)
# # #     except Exception as e:
# # #         print("秘钥错误，解密失败。错误信息：", e)
# # #         return

# # #     # 加载解密后的模型字典
# # #     buffer = io.BytesIO(decrypted_data)
# # #     encrypted_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
# # #     # 对模型权重进行复杂解密
# # #     decrypted_state_dict = {}
# # #     for name, param in encrypted_state_dict.items():
# # #         if isinstance(param, torch.Tensor):
# # #             decrypted_param = complex_decrypt_matrix(param, key)
# # #             decrypted_state_dict[name] = decrypted_param
# # #         else:
# # #             decrypted_state_dict[name] = param  # 未加密的参数直接保存

# # #     # 保存解密后的模型文件
# # #     torch.save(decrypted_state_dict, output_path)
# # #     print(f"模型解密成功并保存到 {output_path}")

# # # # 输入和执行部分
# # # # encrypted_file_path = input("请输入加密的模型文件路径: ")
# # # # original_password = input("请输入解密时的初始密码: ")
# # # # time_str = input("请输入加密时的时间（格式为 YYYYMMDDHHMM）：")
# # # # key = float(input("请输入矩阵加密的key（浮点数）："))
# # # # output_path = input("请输入解密后模型的保存路径: ")

# # # # 输入和执行部分
# # # encrypted_file_path = 'C:/Users/TCTrolong/Desktop/工作/项目/11rvc开发项目/模型加密&解密/加密测试文件/Output/suannai40k.niubi'
# # # #input("请输入加密的模型文件路径: ")
# # # original_password = input("请输入解密时的初始密码: ")
# # # time_str = input("请输入加密时的时间（格式为 YYYYMMDDHHMM）：")
# # # key = float(input("请输入矩阵加密的key（浮点数）："))
# # # output_path = 'C:/Users/TCTrolong/Desktop/工作/项目/11rvc开发项目/模型加密&解密/加密测试文件/解密输出测试'
# # # #input("请输入解密后模型的保存路径: ")

# # # decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)

