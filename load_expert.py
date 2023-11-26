import os
import json

# 构建文件的绝对路径
file_path = 'tmpkw.json'

# 检查文件是否存在
if os.path.exists(file_path):
    print(f"Loading Replay Buffer from {file_path}...")
else:
    print(f"No saved Replay Buffer found at {file_path}. Starting with an empty buffer.")
    # 提前返回或进行其他处理，具体根据需求而定
    exit()

# 读取 JSON 文件
with open(file_path, "r") as file:
    data = json.load(file)

# 打印输出 Defender 的数据
defender_data = data.get("Defender", [])
print("Defender:")
for entry in defender_data:
    print(entry)

# 打印输出 UAV 的数据
uav_data = data.get("UAV", [])
print("UAV:")
for entry in uav_data:
    print(entry)

# 统计变量个数
defender_var_count = len(defender_data)
uav_var_count = len(uav_data)

# 输出变量个数
print(f"\nNumber of Defender variables: {defender_var_count}")
print(f"Number of UAV variables: {uav_var_count}")