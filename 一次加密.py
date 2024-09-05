import datetime

def encrypt_password(original_password):
    # 获取当前时间
    now = datetime.datetime.now()
    time_num = now.strftime('%Y%m%d%H%M')  # 格式化时间为 '202409041715'
    time_num = int(time_num)
    
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

# 示例
original_password = input("请输入模型加密初始密码: ")
print("初始密码是：",original_password)

encrypted_password = encrypt_password(original_password)
print(f"二次加密最终结果: {encrypted_password}")









