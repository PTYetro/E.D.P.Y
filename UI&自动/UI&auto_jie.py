import os
import torch
import datetime
import io
import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
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
        messagebox.showerror("解密失败", f"秘钥错误，解密失败。错误信息：{e}")
        write_to_log(f"解密失败，错误信息：{e}")
        return

    # 加载解密后的模型字典
    buffer = io.BytesIO(decrypted_data)
    encrypted_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
    # 对模型权重进行复杂解密
    decrypted_state_dict, decrypted_params_info = decrypt_model_weights(encrypted_state_dict, key, calculated_encrypted_password)

    if decrypted_state_dict is None:
        # 解密失败，直接返回
        messagebox.showerror("解密失败", "秘钥错误，解密失败。")
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
        messagebox.showinfo("解密成功", f"模型解密成功并保存到 {output_path}")
    except Exception as e:
        print(f"保存解密后的模型文件时发生错误: {e}")
        write_to_log(f"保存解密后的模型文件时发生错误: {e}")
        messagebox.showerror("解密失败", f"保存解密后的模型文件时发生错误: {e}")

def start_decryption(encrypted_file_path, output_path, original_password, time_str, key):
    """启动解密过程"""
    write_to_log(f"------------------------------------------------")  # 分隔符
    write_to_log(f"开始解密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")
    
    decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)

def run_gui():
    """运行图形界面"""
    root = tk.Tk()
    root.title("解密程序")

    tk.Label(root, text="选择加密文件：").grid(row=0, column=0, padx=10, pady=5)
    encrypted_file_entry = tk.Entry(root, width=50)
    encrypted_file_entry.grid(row=0, column=1, padx=10, pady=5)
    tk.Button(root, text="浏览", command=lambda: encrypted_file_entry.insert(0, filedialog.askopenfilename())).grid(row=0, column=2, padx=5)

    tk.Label(root, text="选择输出解密文件路径：").grid(row=1, column=0, padx=10, pady=5)
    output_path_entry = tk.Entry(root, width=50)
    output_path_entry.grid(row=1, column=1, padx=10, pady=5)
    
    # 使用 asksaveasfilename 并在选择文件后插入路径
    def save_output_path():
        file_path = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch Model", "*.pth")])
        if file_path:  # 如果用户选择了路径
            output_path_entry.delete(0, tk.END)  # 清空当前内容
            output_path_entry.insert(0, file_path)  # 插入新路径

    tk.Button(root, text="浏览", command=save_output_path).grid(row=1, column=2, padx=5)

    tk.Label(root, text="输入初始密码：").grid(row=2, column=0, padx=10, pady=5)
    password_entry = tk.Entry(root, show='*', width=50)
    password_entry.grid(row=2, column=1, padx=10, pady=5)

    tk.Label(root, text="输入加密时间 (格式 YYYYMMDDHHMM)：").grid(row=3, column=0, padx=10, pady=5)
    time_entry = tk.Entry(root, show='*',width=50)
    time_entry.grid(row=3, column=1, padx=10, pady=5)

    tk.Label(root, text="输入解密Key：").grid(row=4, column=0, padx=10, pady=5)
    key_entry = tk.Entry(root,show='*', width=50)
    key_entry.grid(row=4, column=1, padx=10, pady=5)

    tk.Button(root, text="开始解密", command=lambda: start_decryption(
        encrypted_file_entry.get(),
        output_path_entry.get(),
        password_entry.get(),
        time_entry.get(),
        float(key_entry.get())
    )).grid(row=5, column=1, pady=20)

    root.mainloop()

if __name__ == "__main__":
    run_gui()




# import os
# import torch
# import datetime
# import io
# import numpy as np
# import pickle
# import tkinter as tk
# from tkinter import filedialog, messagebox
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# from cryptography.hazmat.backends import default_backend
# from cryptography.hazmat.primitives import hashes, padding

# # 获取当前脚本所在目录
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# LOG_FILE = os.path.join(CURRENT_DIR, 'jiemi_log.txt')

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
#         # 不抛出异常，而是返回一个特殊值，表示解密失败
#         return None  

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
#             decrypted_param = decrypt_non_tensor(param, password, key)
#             if decrypted_param is None:
#                 # 如果解密失败，返回 None
#                 print("秘钥错误，解密失败。")
#                 write_to_log("秘钥错误，解密失败。")
#                 return None, None
#             decrypted_state_dict[name] = decrypted_param
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
#         messagebox.showerror("解密失败", f"秘钥错误，解密失败。错误信息：{e}")
#         write_to_log(f"解密失败，错误信息：{e}")
#         return

#     # 加载解密后的模型字典
#     buffer = io.BytesIO(decrypted_data)
#     encrypted_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
#     # 对模型权重进行复杂解密
#     decrypted_state_dict, decrypted_params_info = decrypt_model_weights(encrypted_state_dict, key, calculated_encrypted_password)

#     if decrypted_state_dict is None:
#         # 解密失败，直接返回
#         messagebox.showerror("解密失败", "秘钥错误，解密失败。")
#         return

#     # 保存解密后的模型文件
#     output_dir = os.path.dirname(output_path)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"创建目录: {output_dir}")
#         write_to_log(f"创建目录: {output_dir}")

#     try:
#         torch.save(decrypted_state_dict, output_path)
#         print(f"模型解密成功并保存到 {output_path}")
#         write_to_log(f"解密完成!\n原文件: {encrypted_file_path}\n解密后文件: {output_path}")
#         write_to_log("\n".join(decrypted_params_info))
#         messagebox.showinfo("解密成功", f"模型解密成功并保存到 {output_path}")
#     except Exception as e:
#         print(f"保存解密后的模型文件时发生错误: {e}")
#         write_to_log(f"保存解密后的模型文件时发生错误: {e}")
#         messagebox.showerror("解密失败", f"保存解密后的模型文件时发生错误: {e}")

# def start_decryption(encrypted_file_path, output_path, original_password, time_str, key):
#     """启动解密过程"""
#     write_to_log(f"------------------------------------------------")  # 分隔符
#     write_to_log(f"开始解密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")
    
#     decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)

# def run_gui():
#     """运行图形界面"""
#     root = tk.Tk()
#     root.title("解密程序")

#     tk.Label(root, text="选择加密文件：").grid(row=0, column=0, padx=10, pady=5)
#     encrypted_file_entry = tk.Entry(root, width=50)
#     encrypted_file_entry.grid(row=0, column=1, padx=10, pady=5)
#     tk.Button(root, text="浏览", command=lambda: encrypted_file_entry.insert(0, filedialog.askopenfilename())).grid(row=0, column=2, padx=5)

#     tk.Label(root, text="选择输出解密文件路径：").grid(row=1, column=0, padx=10, pady=5)
#     output_path_entry = tk.Entry(root, width=50)
#     output_path_entry.grid(row=1, column=1, padx=10, pady=5)
#     tk.Button(root, text="浏览", command=lambda: output_path_entry.insert(0, filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch Model", "*.pth")]))).grid(row=1, column=2, padx=5)

#     tk.Label(root, text="输入初始密码：").grid(row=2, column=0, padx=10, pady=5)
#     password_entry = tk.Entry(root, show='*', width=50)
#     password_entry.grid(row=2, column=1, padx=10, pady=5)

#     tk.Label(root, text="输入加密时间 (格式 YYYYMMDDHHMM)：").grid(row=3, column=0, padx=10, pady=5)
#     time_entry = tk.Entry(root, width=50)
#     time_entry.grid(row=3, column=1, padx=10, pady=5)

#     tk.Label(root, text="输入解密Key：").grid(row=4, column=0, padx=10, pady=5)
#     key_entry = tk.Entry(root, width=50)
#     key_entry.grid(row=4, column=1, padx=10, pady=5)

#     tk.Button(root, text="开始解密", command=lambda: start_decryption(
#         encrypted_file_entry.get(),
#         output_path_entry.get(),
#         password_entry.get(),
#         time_entry.get(),
#         float(key_entry.get())
#     )).grid(row=5, column=1, pady=20)

#     root.mainloop()

# if __name__ == "__main__":
#     run_gui()




# import os
# import torch
# import datetime
# import io
# import numpy as np
# import pickle  # 新增导入
# import tkinter as tk
# from tkinter import filedialog, messagebox
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# from cryptography.hazmat.backends import default_backend
# from cryptography.hazmat.primitives import hashes, padding

# # 获取当前脚本所在目录
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# LOG_FILE = os.path.join(CURRENT_DIR, 'jiemi_log.txt')

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
#         # 不抛出异常，而是返回一个特殊值，表示解密失败
#         return None  

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
#             decrypted_param = decrypt_non_tensor(param, password, key)
#             if decrypted_param is None:
#                 # 如果解密失败，返回 None
#                 print("秘钥错误，解密失败。")
#                 write_to_log("秘钥错误，解密失败。")
#                 return None, None
#             decrypted_state_dict[name] = decrypted_param
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
#         messagebox.showerror("解密失败", f"秘钥错误，解密失败。错误信息：{e}")
#         write_to_log(f"解密失败，错误信息：{e}")
#         return

#     # 加载解密后的模型字典
#     buffer = io.BytesIO(decrypted_data)
#     encrypted_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
#     # 对模型权重进行复杂解密
#     decrypted_state_dict, decrypted_params_info = decrypt_model_weights(encrypted_state_dict, key, calculated_encrypted_password)

#     if decrypted_state_dict is None:
#         # 解密失败，直接返回
#         return

#     # 保存解密后的模型文件
#     output_dir = os.path.dirname(output_path)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"创建目录: {output_dir}")
#         write_to_log(f"创建目录: {output_dir}")

#     try:
#         torch.save(decrypted_state_dict, output_path)
#         print(f"模型解密成功并保存到 {output_path}")
#         write_to_log(f"解密完成!\n原文件: {encrypted_file_path}\n解密后文件: {output_path}")
#         write_to_log("\n".join(decrypted_params_info))
#         messagebox.showinfo("解密成功", f"模型解密成功并保存到 {output_path}")
#     except Exception as e:
#         print(f"保存解密后的模型文件时发生错误: {e}")
#         write_to_log(f"保存解密后的模型文件时发生错误: {e}")
#         messagebox.showerror("解密失败", f"保存解密后的模型文件时发生错误: {e}")

# def start_decryption(encrypted_file_path, output_path, original_password, time_str, key):
#     """启动解密过程"""
#     write_to_log(f"------------------------------------------------")  # 分隔符
#     write_to_log(f"开始解密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")
    
#     decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)

# def run_gui():
#     """运行图形界面"""
#     root = tk.Tk()
#     root.title("解密程序")

#     tk.Label(root, text="选择加密文件：").grid(row=0, column=0, padx=10, pady=5)
#     encrypted_file_entry = tk.Entry(root, width=50)
#     encrypted_file_entry.grid(row=0, column=1, padx=10, pady=5)
#     tk.Button(root, text="浏览", command=lambda: encrypted_file_entry.insert(0, filedialog.askopenfilename())).grid(row=0, column=2, padx=5)

#     tk.Label(root, text="选择输出解密文件路径：").grid(row=1, column=0, padx=10, pady=5)
#     output_path_entry = tk.Entry(root, width=50)
#     output_path_entry.grid(row=1, column=1, padx=10, pady=5)
#     tk.Button(root, text="浏览", command=lambda: output_path_entry.insert(0, filedialog.asksaveasfilename(defaultextension=".pth"))).grid(row=1, column=2, padx=5)

#     tk.Label(root, text="输入初始密码：").grid(row=2, column=0, padx=10, pady=5)
#     password_entry = tk.Entry(root, show='*', width=50)
#     password_entry.grid(row=2, column=1, padx=10, pady=5)

#     tk.Label(root, text="输入加密时间 (格式 YYYYMMDDHHMM)：").grid(row=3, column=0, padx=10, pady=5)
#     time_entry = tk.Entry(root, width=50)
#     time_entry.grid(row=3, column=1, padx=10, pady=5)

#     tk.Label(root, text="输入解密Key：").grid(row=4, column=0, padx=10, pady=5)
#     key_entry = tk.Entry(root, width=50)
#     key_entry.grid(row=4, column=1, padx=10, pady=5)

#     tk.Button(root, text="开始解密", command=lambda: start_decryption(
#         encrypted_file_entry.get(),
#         output_path_entry.get(),
#         password_entry.get(),
#         time_entry.get(),
#         float(key_entry.get())
#     )).grid(row=5, column=1, pady=20)

#     root.mainloop()

# if __name__ == "__main__":
#     run_gui()




# import os
# import torch
# import datetime
# import io
# import numpy as np
# import pickle
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# from cryptography.hazmat.backends import default_backend
# from cryptography.hazmat.primitives import hashes, padding
# import tkinter as tk
# from tkinter import filedialog, messagebox

# # 获取当前脚本所在目录
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# # 日志文件路径
# LOG_FILE = os.path.join(CURRENT_DIR, 'decrypt_log.txt')

# def write_to_log(content):
#     """记录运行日志到log.txt文件。"""
#     with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
#         log_file.write(content + '\n')

# def calculate_encrypted_password(original_password, time_str):
#     """根据输入的时间计算二次加密的密码。"""
#     time_num = int(time_str)
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

#     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
#     decryptor = cipher.decryptor()

#     padded_data = decryptor.update(actual_ciphertext) + decryptor.finalize()

#     try:
#         unpadder = padding.PKCS7(128).unpadder()
#         plaintext = unpadder.update(padded_data) + unpadder.finalize()
#     except ValueError as e:
#         write_to_log(f"解密失败，原因：{e}")
#         raise ValueError("解密失败，填充无效！")

#     return plaintext

# def complex_decrypt_tensor(tensor, key, seed):
#     """对模型的权重张量进行解密"""
#     tensor_np = tensor.cpu().numpy()

#     # 使用固定种子恢复顺序
#     np.random.seed(seed)
#     perm = np.random.permutation(tensor_np.size)
#     decrypted_tensor = tensor_np.flatten()[np.argsort(perm)].reshape(tensor_np.shape)

#     noise = np.random.normal(0, 0.1, tensor_np.shape)
#     decrypted_tensor = (decrypted_tensor - noise * key)

#     return torch.tensor(decrypted_tensor, dtype=tensor.dtype, device=tensor.device)

# def decrypt_non_tensor(encrypted_data, password, key):
#     """对非张量参数进行解密"""
#     decrypted_data = decrypt_file_in_memory(encrypted_data, password)
#     key_bytes = int(key * 1000).to_bytes(8, byteorder='big', signed=True)

#     if decrypted_data[:8] != key_bytes:
#         raise ValueError("解密失败，输入的key不正确！")  # 如果输入的key不正确，解密失败

#     serialized_data = decrypted_data[8:]
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
#             decrypted_state_dict[name] = decrypt_non_tensor(param, password, key)
#             decrypted_params_info.append(f"参数 {name} 已解密 - 类型: {type(param)}")
#     return decrypted_state_dict, decrypted_params_info

# def decrypt_model(encrypted_file_path, original_password, time_str, key, output_path):
#     with open(encrypted_file_path, 'rb') as f:
#         encrypted_data = f.read()

#     calculated_encrypted_password = calculate_encrypted_password(original_password, time_str)
#     write_to_log(f"计算出的二次加密密码: {calculated_encrypted_password}")

#     try:
#         decrypted_data = decrypt_file_in_memory(encrypted_data, calculated_encrypted_password)
#     except Exception as e:
#         messagebox.showerror("解密失败", f"秘钥错误，解密失败。错误信息：{e}")
#         write_to_log(f"解密失败，错误信息：{e}")
#         return

#     buffer = io.BytesIO(decrypted_data)
#     model_data = torch.load(buffer, map_location=torch.device('cpu'), weights_only=True)

#     if 'state_dict' not in model_data or 'seed' not in model_data:
#         messagebox.showerror("解密失败", "解密文件格式不正确。")
#         write_to_log("解密失败：解密文件格式不正确。")
#         return

#     encrypted_state_dict = model_data['state_dict']
#     seed = model_data['seed']

#     decrypted_state_dict, decrypted_params_info = decrypt_model_weights(encrypted_state_dict, key, calculated_encrypted_password, seed)

#     output_dir = os.path.dirname(output_path)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         write_to_log(f"创建目录: {output_dir}")

#     try:
#         torch.save(decrypted_state_dict, output_path)
#         messagebox.showinfo("解密成功", f"模型解密成功并保存到 {output_path}")
#         write_to_log(f"解密完成: 原文件: {encrypted_file_path}, 解密后文件: {output_path}")
#         write_to_log("\n".join(decrypted_params_info))
#     except Exception as e:
#         messagebox.showerror("保存错误", f"保存解密后的模型文件时发生错误: {e}")
#         write_to_log(f"保存解密后的模型文件时发生错误: {e}")

# def run_gui():
#     """运行解密程序的图形界面。"""
#     root = tk.Tk()
#     root.title("模型解密程序")

#     def select_encrypted_file():
#         file_path = filedialog.askopenfilename(title="选择要解密的文件")
#         encrypted_file_entry.delete(0, tk.END)
#         encrypted_file_entry.insert(tk.END, file_path)

#     def select_output_path():
#         folder_path = filedialog.askdirectory(title="选择输出文件夹")
#         output_path_entry.delete(0, tk.END)
#         output_path_entry.insert(tk.END, folder_path)

#     def start_decryption(input_file, output_folder, original_password, time_str, key):
#         if not input_file or not output_folder or not original_password or not time_str or not key:
#             messagebox.showerror("错误", "所有输入项都不能为空！")
#             return
        
#         try:
#             key = float(key)
#         except ValueError:
#             messagebox.showerror("错误", "Key 必须是浮点数！")
#             return

#         decrypt_model(input_file, original_password, time_str, key, os.path.join(output_folder, 'decrypted_model.pth'))

#     tk.Label(root, text="选择要解密的文件：").grid(row=0, column=0, padx=10, pady=5)
#     encrypted_file_entry = tk.Entry(root, width=50)
#     encrypted_file_entry.grid(row=0, column=1, padx=10, pady=5)
#     tk.Button(root, text="浏览", command=select_encrypted_file).grid(row=0, column=2, padx=10, pady=5)

#     tk.Label(root, text="选择输出文件夹：").grid(row=1, column=0, padx=10, pady=5)
#     output_path_entry = tk.Entry(root, width=50)
#     output_path_entry.grid(row=1, column=1, padx=10, pady=5)
#     tk.Button(root, text="浏览", command=select_output_path).grid(row=1, column=2, padx=10, pady=5)

#     tk.Label(root, text="输入初始密码：").grid(row=2, column=0, padx=10, pady=5)
#     password_entry = tk.Entry(root, width=50, show='*')
#     password_entry.grid(row=2, column=1, padx=10, pady=5)

#     tk.Label(root, text="输入加密时间 (格式 YYYYMMDDHHMM)：").grid(row=3, column=0, padx=10, pady=5)
#     time_entry = tk.Entry(root, width=50)
#     time_entry.grid(row=3, column=1, padx=10, pady=5)

#     tk.Label(root, text="输入解密Key：").grid(row=4, column=0, padx=10, pady=5)
#     key_entry = tk.Entry(root, width=50)
#     key_entry.grid(row=4, column=1, padx=10, pady=5)

#     tk.Button(root, text="开始解密", command=lambda: start_decryption(
#         encrypted_file_entry.get(),
#         output_path_entry.get(),
#         password_entry.get(),
#         time_entry.get(),
#         key_entry.get()
#     )).grid(row=5, column=1, pady=20)

#     root.mainloop()

# if __name__ == "__main__":
#     run_gui()


# def decrypt_file_in_memory(ciphertext, password):
#     """对文件内容进行 AES-256 解密。"""
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
#     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
#     decryptor = cipher.decryptor()
#     padded_data = decryptor.update(actual_ciphertext) + decryptor.finalize()
#     try:
#         unpadder = padding.PKCS7(128).unpadder()
#         plaintext = unpadder.update(padded_data) + unpadder.finalize()
#     except ValueError as e:
#         write_to_log(f"解密失败，原因：{e}")
#         raise ValueError("解密失败，输入的密码或key不正确！")  # 抛出错误
#     return plaintext

# def complex_decrypt_tensor(tensor, key, seed):
#     """对模型的权重张量进行解密。"""
#     tensor_np = tensor.cpu().numpy()
#     np.random.seed(seed)  # 使用固定种子
#     perm = np.random.permutation(tensor_np.size)
#     decrypted_tensor = tensor_np.flatten()[np.argsort(perm)].reshape(tensor_np.shape)
#     noise = np.random.normal(0, 0.1, tensor_np.shape)
#     decrypted_tensor = (decrypted_tensor - noise * key)
#     return torch.tensor(decrypted_tensor, dtype=tensor.dtype, device=tensor.device)

# def decrypt_non_tensor(encrypted_data, password, key):
#     """对非张量参数进行解密。"""
#     decrypted_data = decrypt_file_in_memory(encrypted_data, password)
#     key_bytes = int(key * 1000).to_bytes(8, byteorder='big', signed=True)
#     if decrypted_data[:8] != key_bytes:
#         raise ValueError("解密失败，输入的key不正确！")
#     value = pickle.loads(decrypted_data[8:])  # 使用pickle反序列化
#     return value

# def decrypt_model_weights(encrypted_state_dict, key, password, seed):
#     """对模型的每个权重矩阵和非张量参数进行复杂解密。"""
#     decrypted_state_dict = {}
#     decrypted_params_info = []  # 用于记录解密的参数信息
#     for name, param in encrypted_state_dict.items():
#         if isinstance(param, torch.Tensor):
#             decrypted_param = complex_decrypt_tensor(param, key, seed)
#             decrypted_state_dict[name] = decrypted_param
#             decrypted_params_info.append(f"参数 {name} 已解密 - 类型: 张量")
#         else:
#             decrypted_state_dict[name] = decrypt_non_tensor(param, password, key)
#             decrypted_params_info.append(f"参数 {name} 已解密 - 类型: {type(param)}")
#     return decrypted_state_dict, decrypted_params_info

# def decrypt_model(encrypted_file_path, original_password, time_str, key, output_path):
#     """解密模型文件并保存解密后的文件。"""
#     with open(encrypted_file_path, 'rb') as f:
#         encrypted_data = f.read()
#     calculated_encrypted_password = calculate_encrypted_password(original_password, time_str)
#     write_to_log(f"计算出的二次加密密码: {calculated_encrypted_password}")
#     try:
#         decrypted_data = decrypt_file_in_memory(encrypted_data, calculated_encrypted_password)
#     except Exception as e:
#         messagebox.showerror("解密失败", f"解密失败，错误信息：{e}")
#         return
#     buffer = io.BytesIO(decrypted_data)
#     model_data = torch.load(buffer, map_location=torch.device('cpu'))
#     encrypted_state_dict = model_data['state_dict']
#     seed = model_data['seed']
#     decrypted_state_dict, decrypted_params_info = decrypt_model_weights(encrypted_state_dict, key, calculated_encrypted_password, seed)
#     output_dir = os.path.dirname(output_path)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         write_to_log(f"创建目录: {output_dir}")
#     try:
#         torch.save(decrypted_state_dict, output_path)
#         write_to_log(f"解密完成: 原文件: {encrypted_file_path}, 解密后文件: {output_path}")
#         write_to_log("\n".join(decrypted_params_info))
#         messagebox.showinfo("解密成功", f"模型解密成功并保存到 {output_path}")
#     except Exception as e:
#         messagebox.showerror("错误", f"保存解密后的模型文件时发生错误: {e}")

# # 图形界面部分
# def browse_file(entry):
#     """浏览文件对话框来选择加密的模型文件。"""
#     file_path = filedialog.askopenfilename(title="选择加密的模型文件", filetypes=(("加密文件", "*.niubi"), ("所有文件", "*.*")))
#     entry.delete(0, tk.END)
#     entry.insert(0, file_path)

# def browse_directory(entry):
#     """浏览文件夹对话框来选择输出目录。"""
#     directory = filedialog.askdirectory(title="选择输出目录")
#     entry.delete(0, tk.END)
#     entry.insert(0, directory)

# def start_decryption(encrypted_file_path, output_path, original_password, time_str, key):
#     """开始解密过程的函数。"""
#     try:
#         key = float(key)
#     except ValueError:
#         messagebox.showerror("输入错误", "请输入有效的加密Key（浮点数）。")
#         return
#     write_to_log(f"开始解密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")
#     write_to_log(f"--------------------------")  # 分隔符
#     decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)

# def run_gui():
#     """运行图形用户界面。"""
#     root = tk.Tk()
#     root.title("模型解密程序")

#     # 加密文件选择
#     tk.Label(root, text="选择加密文件：").grid(row=0, column=0, padx=10, pady=5)
#     encrypted_file_entry = tk.Entry(root, width=50)
#     encrypted_file_entry.grid(row=0, column=1, padx=10, pady=5)
#     tk.Button(root, text="浏览", command=lambda: browse_file(encrypted_file_entry)).grid(row=0, column=2, padx=10, pady=5)

#     # 输出目录选择
#     tk.Label(root, text="选择输出目录：").grid(row=1, column=0, padx=10, pady=5)
#     output_path_entry = tk.Entry(root, width=50)
#     output_path_entry.grid(row=1, column=1, padx=10, pady=5)
#     tk.Button(root, text="浏览", command=lambda: browse_directory(output_path_entry)).grid(row=1, column=2, padx=10, pady=5)

#     # 初始密码输入
#     tk.Label(root, text="输入初始密码：").grid(row=2, column=0, padx=10, pady=5)
#     password_entry = tk.Entry(root, width=50, show='*')
#     password_entry.grid(row=2, column=1, padx=10, pady=5)

#     # 加密时间输入
#     tk.Label(root, text="输入加密时间（格式 YYYYMMDDHHMM）：").grid(row=3, column=0, padx=10, pady=5)
#     time_entry = tk.Entry(root, width=50)
#     time_entry.grid(row=3, column=1, padx=10, pady=5)

#     # 解密Key输入
#     tk.Label(root, text="输入解密Key：").grid(row=4, column=0, padx=10, pady=5)
#     key_entry = tk.Entry(root, width=50, show='*')
#     key_entry.grid(row=4, column=1, padx=10, pady=5)

#     # 解密按钮
#     tk.Button(root, text="开始解密", command=lambda: start_decryption(
#         encrypted_file_entry.get(),
#         output_path_entry.get(),
#         password_entry.get(),
#         time_entry.get(),
#         key_entry.get()
#     )).grid(row=5, column=1, pady=20)

#     root.mainloop()

# # 启动图形界面
# if __name__ == "__main__":
#     run_gui()




# import os
# import torch
# import io
# import numpy as np
# import pickle
# import datetime
# import tkinter as tk
# from tkinter import filedialog, messagebox
# from tkinter import ttk
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
#         messagebox.showerror("解密失败", f"解密失败，错误信息：{e}")  # 添加解密失败提示
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
#         messagebox.showinfo("解密成功", f"模型解密成功并保存到 {output_path}")
#     except Exception as e:
#         print(f"保存解密后的模型文件时发生错误: {e}")
#         write_to_log(f"保存解密后的模型文件时发生错误: {e}")
#         messagebox.showerror("错误", f"保存解密后的模型文件时发生错误: {e}")

# # 图形用户界面部分
# def select_file():
#     """选择需要解密的文件"""
#     file_path = filedialog.askopenfilename(title="选择加密文件", filetypes=[("Encrypted Files", "*.niubi"), ("All Files", "*.*")])
#     if file_path:
#         encrypted_file_path_var.set(file_path)

# def select_output_path():
#     """选择解密输出路径"""
#     dir_path = filedialog.askdirectory(title="选择解密输出目录")
#     if dir_path:
#         output_path_var.set(os.path.join(dir_path, "解密后的文件.pth"))

# def start_decryption():
#     """开始解密"""
#     encrypted_file_path = encrypted_file_path_var.get()
#     original_password = password_entry.get()
#     time_str = time_entry.get()
#     key = key_entry.get()
#     output_path = output_path_var.get()

#     if not (encrypted_file_path and original_password and time_str and key and output_path):
#         messagebox.showerror("错误", "所有字段都是必填项！")
#         return

#     try:
#         key = float(key)
#     except ValueError:
#         messagebox.showerror("错误", "Key 必须是一个浮点数！")
#         return

#     write_to_log(f"开始解密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")

#     decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)

# # 创建主窗口
# root = tk.Tk()
# root.title("解密程序")

# # 输入加密文件路径
# ttk.Label(root, text="加密文件路径:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
# encrypted_file_path_var = tk.StringVar()
# ttk.Entry(root, textvariable=encrypted_file_path_var, width=40).grid(row=0, column=1, padx=10, pady=10)
# ttk.Button(root, text="选择文件", command=select_file).grid(row=0, column=2, padx=10, pady=10)

# # 输入初始密码
# ttk.Label(root, text="初始密码:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
# password_entry = ttk.Entry(root, show="*")
# password_entry.grid(row=1, column=1, padx=10, pady=10)

# # 输入加密时间
# ttk.Label(root, text="加密时间 (格式: YYYYMMDDHHMM):").grid(row=2, column=0, padx=10, pady=10, sticky="w")
# time_entry = ttk.Entry(root, show="*")  # 使用 show="*" 隐藏输入
# time_entry.grid(row=2, column=1, padx=10, pady=10)

# # 输入矩阵加密的key
# ttk.Label(root, text="矩阵加密的key (浮点数):").grid(row=3, column=0, padx=10, pady=10, sticky="w")
# key_entry = ttk.Entry(root, show="*")  # 使用 show="*" 隐藏输入
# key_entry.grid(row=3, column=1, padx=10, pady=10)

# # 解密输出路径
# ttk.Label(root, text="解密输出路径:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
# output_path_var = tk.StringVar()
# ttk.Entry(root, textvariable=output_path_var, width=40).grid(row=4, column=1, padx=10, pady=10)
# ttk.Button(root, text="选择输出路径", command=select_output_path).grid(row=4, column=2, padx=10, pady=10)

# # 解密按钮
# ttk.Button(root, text="开始解密", command=start_decryption).grid(row=5, column=0, columnspan=3, padx=10, pady=20)

# root.mainloop()



# import os
# import torch
# import io
# import numpy as np
# import pickle
# import datetime
# import tkinter as tk
# from tkinter import filedialog, messagebox
# from tkinter import ttk
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
#         messagebox.showinfo("解密成功", f"模型解密成功并保存到 {output_path}")
#     except Exception as e:
#         print(f"保存解密后的模型文件时发生错误: {e}")
#         write_to_log(f"保存解密后的模型文件时发生错误: {e}")
#         messagebox.showerror("错误", f"保存解密后的模型文件时发生错误: {e}")

# # 图形用户界面部分
# def select_file():
#     """选择需要解密的文件"""
#     file_path = filedialog.askopenfilename(title="选择加密文件", filetypes=[("Encrypted Files", "*.niubi"), ("All Files", "*.*")])
#     if file_path:
#         encrypted_file_path_var.set(file_path)

# def select_output_path():
#     """选择解密输出路径"""
#     dir_path = filedialog.askdirectory(title="选择解密输出目录")
#     if dir_path:
#         output_path_var.set(os.path.join(dir_path, "解密后的文件.pth"))

# def start_decryption():
#     """开始解密"""
#     encrypted_file_path = encrypted_file_path_var.get()
#     original_password = password_entry.get()
#     time_str = time_entry.get()
#     key = key_entry.get()
#     output_path = output_path_var.get()

#     if not (encrypted_file_path and original_password and time_str and key and output_path):
#         messagebox.showerror("错误", "所有字段都是必填项！")
#         return

#     try:
#         key = float(key)
#     except ValueError:
#         messagebox.showerror("错误", "Key 必须是一个浮点数！")
#         return

#     write_to_log(f"开始解密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")
#     write_to_log(f"--------------------------")  # 分隔线
#     decrypt_model(encrypted_file_path, original_password, time_str, key, output_path)

# # 创建主窗口
# root = tk.Tk()
# root.title("解密程序")

# # 输入加密文件路径
# ttk.Label(root, text="加密文件路径:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
# encrypted_file_path_var = tk.StringVar()
# ttk.Entry(root, textvariable=encrypted_file_path_var, width=40).grid(row=0, column=1, padx=10, pady=10)
# ttk.Button(root, text="选择文件", command=select_file).grid(row=0, column=2, padx=10, pady=10)

# # 输入初始密码
# ttk.Label(root, text="初始密码:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
# password_entry = ttk.Entry(root, show="*")
# password_entry.grid(row=1, column=1, padx=10, pady=10)

# # 输入加密时间
# ttk.Label(root, text="加密时间 (格式: YYYYMMDDHHMM):").grid(row=2, column=0, padx=10, pady=10, sticky="w")
# time_entry = ttk.Entry(root)
# time_entry.grid(row=2, column=1, padx=10, pady=10)

# # 输入矩阵加密的key
# ttk.Label(root, text="矩阵加密的key (浮点数):").grid(row=3, column=0, padx=10, pady=10, sticky="w")
# key_entry = ttk.Entry(root)
# key_entry.grid(row=3, column=1, padx=10, pady=10)

# # 解密输出路径
# ttk.Label(root, text="解密输出路径:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
# output_path_var = tk.StringVar()
# ttk.Entry(root, textvariable=output_path_var, width=40).grid(row=4, column=1, padx=10, pady=10)
# ttk.Button(root, text="选择输出路径", command=select_output_path).grid(row=4, column=2, padx=10, pady=10)

# # 解密按钮
# ttk.Button(root, text="开始解密", command=start_decryption).grid(row=5, column=0, columnspan=3, padx=10, pady=20)

# root.mainloop()
