import os
import csv
from ultralytics import YOLO
from mmpose.apis import MMPoseInferencer
from mmpose.utils import register_all_modules

register_all_modules()

input_file_dir = "../KABR/dataset/image"  # 待检测图片的文件夹
out_file_dir = "out"  # 检测结果保存的文件夹
#字典对应名字和数据真实值
animals_dict = {'ZG': 'grevy-zebra', 'ZP': 'great-zebra', 'G0': 'giraffe'}
#acc1
accnum = 0
allnum = 0
#acc2
TP = 0
TN = 0
FP = 0
FN = 0 

# Load detection model
model_detection = YOLO("yolov8s.yaml")
model_detection = YOLO("runs/detect/train2/weights/best.pt")

# Load keypoint detection model
model_pose = MMPoseInferencer(
    pose2d="mmpose/configs/animal_2d_keypoint/rtmpose/ap10k/rtmpose-m_8xb64-210e_ap10k-256x256.py",
    pose2d_weights="grevy_zebra.pth",
)


# 打开 CSV 文件
with open('../KABR/annotation/detecttest.csv', newline='') as csvfile:
    # 创建 CSV 读取器
    reader = csv.reader(csvfile, delimiter=' ')
    # 读取标题行
    headers = next(reader)[0].split()  # 按空格分割列名
   
    # 用于存储提取的数据
    extracted_folder = []
    extracted_data = []
    # 逐行读取数据
    for row in reader:
        # 提取 path
        folder = row[0]

     #   vid = row[1]
     #   fid = row[2]
        path = row[3]
        # 将提取的数据添加到列表中
     #   extracted_folder.append(folder)
        
        truthclass = folder[:2]
        data = [path,truthclass]
        extracted_data.append(data)

# Predict with the model
#for file in os.listdir(input_file_dir):  # 遍历待检测图片的文件夹下的所有图片
for data in extracted_data:     
    file_path = os.path.join(input_file_dir, data[0])
    modified_path = data[0].replace('/', '_')
    results = model_detection(
        file_path, conf=0.75 # box 目标检测的阈值
    )  # 使用目标检测模型检测图像为哪一类动物，conf是模型的置信度，低于这个值则认为没有检测到结果


    for result in results:
        print("Now file path is ", result.path)

        detection_pred_label = [
            result.names[int(i.cpu().numpy())] for i in result.boxes.cls
        ]  # 处理标签，将目标检测模型预测的标签从索引转成文字
#check accurency1
        classname = animals_dict[data[1]]
        if classname in detection_pred_label:
            accnum = accnum + 1
        #输出分类结果
        result.plot(
                kpt_radius=5, save=True, filename=f"detectoutput/{modified_path}"
            )  # 将检测模型的结果保存至temp文件夹下
        allnum = allnum + 1

#check accurency2 Binary Classification  (TP + TN) / (TP + TN + FP + FN)
        #True Positive (TP)：正确预测为正类别的样本数量
        if "grevy-zebra" in detection_pred_label and classname == "grevy-zebra":
            TP = TP + 1
        #False Positive (FP)：错误预测为正类别的样本数量。
        if "grevy-zebra" in detection_pred_label and classname != "grevy-zebra":
            FP = FP + 1
        #True Negative (TN)：正确预测为负类别的样本数量。
        if "grevy-zebra" not in detection_pred_label and classname != "grevy-zebra":
            TN = TN + 1
        #False Negative (FN)：错误预测为负类别的样本数量。
        if "grevy-zebra" not in detection_pred_label and classname == "grevy-zebra":
            FN = FN + 1

#仅grevy-zebra进行动作提取
        if "grevy-zebra" in detection_pred_label:
            result.plot(
                kpt_radius=4, save=True, filename=f"temp/{modified_path}"
            )  # 将检测模型的结果保存至temp文件夹下
            result_generator = model_pose(
                f"temp/{modified_path}",
                vis_out_dir=f"{out_file_dir}/{modified_path}",
                radius=4,
                thickness=1,
                # draw_bbox=True,
                kpt_thr=0.5, # keypoint 阈值
                return_vis=True,
            )  # 使用关键点检测模型检测关键点

            result = next(result_generator)

        else:
            result.plot(
                kpt_radius=4, save=True, filename=f"{out_file_dir}/{modified_path}"
            )  # 将结果保存至结果文件夹下
#acc1
#print(accnum/allnum)
#acc2
print((TP + TN) / (TP + TN + FP + FN))