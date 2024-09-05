*`Encryption and Decryption of Pytorch model files by YetroTech`*

需要安装cryptography以进行AES对称加密
终端运行以下：
pip install cryptography

若未安装PyTorch库，需要进行安装
终端运行以下：
pip install torch

！！注意！！
若需要传入文件绝对路径，必须使用正斜杠/    ,     而不是反斜杠\

一共四层加密
1.管理员在上传模型的时候输入对该模型加密的初始密码
2.密码加密程序对初始密码根据模型上传的时间进行二次加密，得到二次密码
3.二次密码进行加密获得三次密码
4.加密程序使用三次密码对模型进行加密

三次密码的强度极高，有效防止用户恶意破解。

二次加密思路：

需要读取此时的电脑时间。
假如当前时间为2024.9.4.17.15（2024年9月4日17点15分），得到数字编码202409041715（用time_num命名），
假如初始密码为123456（用original_password命名），对初始密码每个字符进行ASCII的转化，即495051525354
（用original_password_ascii命名），数字编码进行数学乘二得404818083430（用time_numx2命名），字符
ASCII值（original_password_ascii）进行数学乘三得1485154576062（用original_password_asciix3命名），
对这两个数字进行数学加法运算，得到1889972659492（用add_num命名），将add_num从末尾开始每两位划分为一组，
并倒序排列，得到int类数组[92,94,65,72,99,88,1]（用array_partitioning命名），分别对应为ASCII值，得到
char类字符组[\,^,A,H,c,X,1]（用char_group命名），将char_group分别间隔个数字插入add_num中（若末尾数字
个数不够，则在完成最后一次合法插入后放置剩余数字，在剩余数字后直接添加char_group中剩余字符），得到二次
加密最终结果，用second_encryption命名。

三次加密思路：
MD5->base64（暂未实现，有待更新）

使用第三次加密之后的密码对模型文件进行AES对称加密。