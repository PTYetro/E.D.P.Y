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
import tkinter as tk
from tkinter import filedialog, messagebox

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

def run_gui():
    """
    运行加密程序的图形界面。
    """
    root = tk.Tk()
    root.title("模型加密程序")

    # 加密文件选择
    def select_encrypted_file():
        file_path = filedialog.askopenfilename(title="选择要加密的文件")
        encrypted_file_entry.delete(0, tk.END)
        encrypted_file_entry.insert(tk.END, file_path)

    # 输出路径选择
    def select_output_path():
        folder_path = filedialog.askdirectory(title="选择输出文件夹")
        output_path_entry.delete(0, tk.END)
        output_path_entry.insert(tk.END, folder_path)

    # 开始加密按钮点击事件
    def start_encryption(input_file, output_folder, original_password, key):
        if not input_file or not output_folder or not original_password or not key:
            messagebox.showerror("错误", "所有输入项都不能为空！")
            return
        
        try:
            key = float(key)
        except ValueError:
            messagebox.showerror("错误", "Key 必须是浮点数！")
            return
        
        encrypted_password, time_num = encrypt_password(original_password)
        password = encrypted_password
        
        write_to_log(f"开始加密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")
        write_to_log(f"---------------------------------------------------")  # 分隔符

        encrypted_file_path = encrypt_model(input_file, password, key, output_extension='.niubi', output_dir=output_folder)

        run_count = update_run_count()
        info_content = f"运行次数: {run_count}\n文件名: {os.path.basename(input_file)}\n文件路径: {input_file}\n初始密码: {original_password}\n二次加密后的密码: {encrypted_password}\n加密时间: {time_num}\nkey: {key}\n加密后文件的输出路径: {encrypted_file_path}\n--------------------------"
        write_to_info(info_content)

        messagebox.showinfo("成功", f"模型已加密并保存到 {encrypted_file_path}")

    # 创建界面元素
    tk.Label(root, text="选择要加密的文件：").grid(row=0, column=0, padx=10, pady=5)
    encrypted_file_entry = tk.Entry(root, width=50)
    encrypted_file_entry.grid(row=0, column=1, padx=10, pady=5)
    tk.Button(root, text="浏览", command=select_encrypted_file).grid(row=0, column=2, padx=10, pady=5)

    tk.Label(root, text="选择输出文件夹：").grid(row=1, column=0, padx=10, pady=5)
    output_path_entry = tk.Entry(root, width=50)
    output_path_entry.grid(row=1, column=1, padx=10, pady=5)
    tk.Button(root, text="浏览", command=select_output_path).grid(row=1, column=2, padx=10, pady=5)

    tk.Label(root, text="输入初始密码：").grid(row=2, column=0, padx=10, pady=5)
    password_entry = tk.Entry(root, width=50, show='*')
    password_entry.grid(row=2, column=1, padx=10, pady=5)

    tk.Label(root, text="输入加密Key：").grid(row=3, column=0, padx=10, pady=5)
    key_entry = tk.Entry(root, width=50,show='*')
    key_entry.grid(row=3, column=1, padx=10, pady=5)

    tk.Button(root, text="开始加密", command=lambda: start_encryption(
        encrypted_file_entry.get(),
        output_path_entry.get(),
        password_entry.get(),
        key_entry.get()
    )).grid(row=4, column=1, pady=20)

    root.mainloop()

# 启动图形界面
if __name__ == "__main__":
    run_gui()



# import os
# import datetime
# import torch
# import io
# import numpy as np
# import pickle  # 用于序列化和反序列化
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# from cryptography.hazmat.backends import default_backend
# from cryptography.hazmat.primitives import hashes, padding
# import tkinter as tk
# from tkinter import filedialog, messagebox

# # 获取当前脚本所在目录
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# # 日志文件和信息文件的路径
# LOG_FILE = os.path.join(CURRENT_DIR, 'jiami_log.txt')
# INFO_FILE = os.path.join(CURRENT_DIR, 'jiami_info.txt')

# def write_to_log(content):
#     """记录运行日志到log.txt文件。"""
#     with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
#         log_file.write(content + '\n')

# def write_to_info(content):
#     """记录加密信息到info.txt文件。"""
#     with open(INFO_FILE, 'a', encoding='utf-8') as info_file:
#         info_file.write(content + '\n')

# def read_run_count():
#     """读取当前运行次数，如果没有记录文件则初始化为0。"""
#     try:
#         with open(INFO_FILE, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
#             return len([line for line in lines if line.startswith('运行次数:')])
#     except FileNotFoundError:
#         return 0

# def update_run_count():
#     """更新当前运行次数。"""
#     run_count = read_run_count() + 1
#     return run_count

# def encrypt_password(original_password):
#     """基于初始密码和当前时间生成二次加密密码。"""
#     now = datetime.datetime.now()
#     time_num = now.strftime('%Y%m%d%H%M')  # 格式化时间为 'YYYYMMDDHHMM'
#     time_num = int(time_num)

#     # 计算加密相关的数字
#     time_numx2 = time_num * 2
#     original_password_ascii = ''.join([str(ord(c)) for c in original_password])
#     original_password_asciix3 = int(original_password_ascii) * 3
#     add_num = time_numx2 + original_password_asciix3

#     # 生成字符组
#     add_num_str = str(add_num)
#     array_partitioning = [int(add_num_str[i:i+2]) for i in range(0, len(add_num_str), 2)][::-1]
#     char_group = [chr(x) for x in array_partitioning]

#     # 插入字符组并生成二次加密密码
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

# def encrypt_file_in_memory(plaintext, password):
#     """使用AES-256对数据进行加密。"""
#     salt = os.urandom(16)  # 随机生成盐
#     kdf = PBKDF2HMAC(
#         algorithm=hashes.SHA256(),
#         length=32,
#         salt=salt,
#         iterations=100000,
#         backend=default_backend()
#     )
#     key = kdf.derive(password.encode())  # 生成密钥

#     # 初始化AES加密算法
#     iv = os.urandom(16)  # 生成随机初始向量
#     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
#     encryptor = cipher.encryptor()

#     # 对数据进行PKCS7填充
#     padder = padding.PKCS7(128).padder()
#     padded_data = padder.update(plaintext) + padder.finalize()

#     # 加密数据
#     ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    
#     return salt + iv + ciphertext

# def complex_encrypt_tensor(tensor, key):
#     """对张量进行加密，添加噪声并打乱顺序。"""
#     tensor_np = tensor.cpu().numpy()
#     noise = np.random.normal(0, 0.1, tensor_np.shape)  # 生成随机噪声
#     encrypted_tensor = tensor_np + noise * key  # 添加噪声

#     # 打乱张量中的元素顺序
#     perm = np.random.permutation(encrypted_tensor.size)
#     encrypted_tensor = encrypted_tensor.flatten()[perm].reshape(tensor_np.shape)
    
#     return torch.tensor(encrypted_tensor, dtype=tensor.dtype, device=tensor.device)

# def encrypt_non_tensor(value, password, key):
#     """对非张量类型参数进行加密。"""
#     serialized_data = pickle.dumps(value)  # 使用pickle序列化为字节
    
#     # 将 key 融入到加密过程中
#     key_bytes = int(key * 1000).to_bytes(8, byteorder='big', signed=True)
#     serialized_data_with_key = key_bytes + serialized_data  # 将 key 作为数据的一部分
    
#     encrypted_data = encrypt_file_in_memory(serialized_data_with_key, password)
#     return encrypted_data

# def encrypt_model_weights(model_state_dict, key, password):
#     """对模型的每个权重矩阵和非张量参数进行复杂加密。"""
#     encrypted_state_dict = {}
#     encrypted_params_info = []  # 用于记录加密的参数信息
#     for name, param in model_state_dict.items():
#         if isinstance(param, torch.Tensor):
#             # 如果是张量，则进行张量加密
#             encrypted_param = complex_encrypt_tensor(param, key)
#             encrypted_state_dict[name] = encrypted_param
#             encrypted_params_info.append(f"参数 {name} 已加密 - 类型: 张量")
#         else:
#             # 加密非张量类型参数，并包含 key 的影响
#             encrypted_state_dict[name] = encrypt_non_tensor(param, password, key)
#             encrypted_params_info.append(f"参数 {name} 已加密 - 类型: {type(param)}")
#     return encrypted_state_dict, encrypted_params_info

# def encrypt_model(file_path, password, key, output_extension='.enc', output_dir=None):
#     """加密模型文件并保存加密后的文件。"""
#     # 加载模型状态字典
#     state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    
#     # 加密模型的权重和参数
#     encrypted_state_dict, encrypted_params_info = encrypt_model_weights(state_dict, key, password)

#     # 将加密后的状态字典写入内存缓冲区
#     buffer = io.BytesIO()
#     torch.save(encrypted_state_dict, buffer)
#     buffer.seek(0)

#     # 加密内存缓冲区的内容
#     plaintext = buffer.read()
#     ciphertext = encrypt_file_in_memory(plaintext, password)

#     # 设置输出路径
#     if output_dir is None:
#         output_dir = os.path.dirname(file_path)
#     else:
#         os.makedirs(output_dir, exist_ok=True)
    
#     base_name = os.path.splitext(os.path.basename(file_path))[0]
#     encrypted_file_path = os.path.join(output_dir, base_name + output_extension)
    
#     # 保存加密后的文件
#     with open(encrypted_file_path, 'wb') as f:
#         f.write(ciphertext)

#     # 写入日志
#     write_to_log(f"加密完成: 原文件: {file_path}, 加密后文件: {encrypted_file_path}")
#     write_to_log("\n".join(encrypted_params_info))
#     return encrypted_file_path

# def start_encryption(model_path, output_path, original_password, key):
#     """开始加密过程的函数。"""
#     password = encrypt_password(original_password)[0]
    
#     # 写入日志
#     write_to_log(f"开始加密: 时间: {datetime.datetime.now()}, 初始密码: {original_password}, key: {key}")
#     write_to_log(f"---------------------------------------------------")  # 分隔符

#     # 执行加密过程
#     encrypted_file_path = encrypt_model(model_path, password, key, output_extension='.niubi', output_dir=output_path)

#     # 记录 info.txt 信息
#     run_count = update_run_count()
#     info_content = f"运行次数: {run_count}\n文件名: {os.path.basename(model_path)}\n文件路径: {model_path}\n初始密码: {original_password}\n二次加密后的密码: {password}\nkey: {key}\n加密后文件的输出路径: {encrypted_file_path}\n--------------------------"
#     write_to_info(info_content)

#     messagebox.showinfo("加密完成", f"模型已加密并保存到 {encrypted_file_path}")

# # 图形界面部分
# def browse_file(entry):
#     """浏览文件对话框来选择模型文件。"""
#     file_path = filedialog.askopenfilename(title="选择模型文件", filetypes=(("PyTorch模型文件", "*.pth"), ("所有文件", "*.*")))
#     entry.delete(0, tk.END)
#     entry.insert(0, file_path)

# def browse_directory(entry):
#     """浏览文件夹对话框来选择输出目录。"""
#     directory = filedialog.askdirectory(title="选择输出目录")
#     entry.delete(0, tk.END)
#     entry.insert(0, directory)

# def run_gui():
#     """运行图形用户界面。"""
#     root = tk.Tk()
#     root.title("模型加密程序")

#     # 模型文件选择
#     tk.Label(root, text="选择模型文件：").grid(row=0, column=0, padx=10, pady=5)
#     model_path_entry = tk.Entry(root, width=50)
#     model_path_entry.grid(row=0, column=1, padx=10, pady=5)
#     tk.Button(root, text="浏览", command=lambda: browse_file(model_path_entry)).grid(row=0, column=2, padx=10, pady=5)

#     # 输出目录选择
#     tk.Label(root, text="选择输出目录：").grid(row=1, column=0, padx=10, pady=5)
#     output_path_entry = tk.Entry(root, width=50)
#     output_path_entry.grid(row=1, column=1, padx=10, pady=5)
#     tk.Button(root, text="浏览", command=lambda: browse_directory(output_path_entry)).grid(row=1, column=2, padx=10, pady=5)

#     # 初始密码输入
#     tk.Label(root, text="输入初始密码：").grid(row=2, column=0, padx=10, pady=5)
#     password_entry = tk.Entry(root, width=50, show='*')
#     password_entry.grid(row=2, column=1, padx=10, pady=5)

#     # 加密Key输入
#     tk.Label(root, text="输入加密Key：").grid(row=3, column=0, padx=10, pady=5)
#     key_entry = tk.Entry(root, width=50, show='*')
#     key_entry.grid(row=3, column=1, padx=10, pady=5)

#     # 加密按钮
#     tk.Button(root, text="开始加密", command=lambda: start_encryption(
#         model_path_entry.get(),
#         output_path_entry.get(),
#         password_entry.get(),
#         float(key_entry.get())
#     )).grid(row=4, column=1, pady=20)

#     root.mainloop()

# # 启动图形界面
# if __name__ == "__main__":
#     run_gui()
