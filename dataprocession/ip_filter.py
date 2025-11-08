import pandas as pd

# 定义输入和输出文件路径
input_file_path = 'data_csv/merged_data_caili.csv'
output_file_path = '../filtered_data_caili.csv'

# 读取CSV文件
df = pd.read_csv(input_file_path)

# 筛选出“IP地址”列的值为"江西"的所有行
# 注意：确保'IP地址'是你的列名，如果不是，请替换为正确的列名
filtered_df = df[df['IP地址'] == '江西']

# 将筛选后的结果保存到新的CSV文件
filtered_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

print(f"已成功将符合条件的数据保存到 {output_file_path}")