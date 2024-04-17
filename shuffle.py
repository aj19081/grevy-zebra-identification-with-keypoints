import csv
import random

# 读取原始CSV文件
with open('../KABR/annotation/idtrain.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

# 打乱数据
random.shuffle(data)

# 将打乱后的数据写入新的CSV文件
with open('../KABR/annotation/shuffled.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

print("CSV文件行已打乱并写入shuffled.csv文件。")
