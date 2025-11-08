import os
import pandas as pd

# 设置文件夹路径
folder_path = '../data'  # 替换为你的文件夹路径

# 获取文件夹中的所有Excel文件
excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

# 创建一个空的DataFrame来存储所有数据
all_data = pd.DataFrame()

# 遍历所有Excel文件并读取数据
for file in excel_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_excel(file_path)
    all_data = pd.concat([all_data, df], ignore_index=True)

# 将合并后的数据保存到一个CSV文件中
output_file = 'data_csv/merged_data_caili.csv'
all_data.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"所有数据已合并并保存到 {output_file}")