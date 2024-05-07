import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import csv
import json
from mmpose.apis import MMPoseInferencer
from mmpose.utils import register_all_modules
import numpy as np
from torch.utils.tensorboard import SummaryWriter

input_file_dir = "../KABR/dataset/image"  # 待检测图片的文件夹
out_file_dir = "idout"  # id检测结果保存的文件夹
iddata = []
testdata = []
valdata = []
# 设定一些超参数
input_size = (100,100)
batch_size = 100
num_epochs = 30
early_stopping_patience = 5
early_stopping_counter = 0
best_val_acc = 0
writer = SummaryWriter('id/experiment_cnn_kp_no_dropout_E')
# Load keypoint detection model
model_pose = MMPoseInferencer(
    pose2d="mmpose/configs/animal_2d_keypoint/rtmpose/ap10k/rtmpose-m_8xb64-210e_ap10k-256x256.py",
    pose2d_weights="grevy_zebra.pth",
)

transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor()
])
def calculate_bbox_area(bbox):
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min) * (y_max - y_min)

# 自定义数据集
class SiameseDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
    
    def __len__(self):
        return len(self.image_folder)
    
    def __getitem__(self, idx):
        file_path1, keypoints1, file_path2, keypoints2, label = self.image_folder[idx]
        img1 = Image.open(file_path1).convert('RGB')
        img2 = Image.open(file_path2).convert('RGB')
        

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, keypoints1, img2, keypoints2, torch.from_numpy(np.array([label], dtype=np.float32))


# 读取CSV文件
image_paths = []
labels = []

# 打开CSV文件进行读取
#ZG0007 - ZG0312
with open('../KABR/annotation/idtrain1.csv', "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    
    # 逐行读取数据
    for row in csv_reader:
        # 获取图像路径和标签
        image1_path, image2_path, label = row
        #目录名
        file_name1 = image1_path.split('/')[0]
        file_name2 = image2_path.split('/')[0]
        #总路径
        file_path1 = os.path.join(input_file_dir, image1_path)
        file_path2 = os.path.join(input_file_dir, image2_path)
    # save keypoints as json file  ZG0007.3/1.json
        '''   
        if not os.path.exists(f"predictions/{image1_path}"):
            result_generator1 = model_pose(file_path1, pred_out_dir=f"predictions/{file_name1}")
            result = next(result_generator1)
        if not os.path.exists(f"predictions/{image2_path}"):
            result_generator2 = model_pose(file_path2, pred_out_dir=f"predictions/{file_name2}")
            result = next(result_generator2)''' 

 # 生成的json文件读取
        # 将文件路径拆分为目录和文件名
        directory, filename = os.path.split(image1_path)
        # 将文件名的扩展名更改为 .json
        new_filename = os.path.splitext(filename)[0] + ".json"
        # 构造新的文件路径
        new_path1 = os.path.join(directory, new_filename)

        directory, filename = os.path.split(image2_path)

        new_filename = os.path.splitext(filename)[0] + ".json"

        new_path2 = os.path.join(directory, new_filename)

        with open(f"predictions/{new_path1}", "r") as json_file1:
            data1 = json.load(json_file1)

        bbox_list1 = [item["bbox"][0] for item in data1]
        bbox_areas1 = [calculate_bbox_area(bbox) for bbox in bbox_list1]
        max_area_index1 = bbox_areas1.index(max(bbox_areas1))
        keypointslist1 = data1[max_area_index1]['keypoints']

        keypoints1 = torch.tensor(keypointslist1, dtype=torch.float32)


        with open(f"predictions/{new_path2}", "r") as json_file2:
            data2 = json.load(json_file2)

        bbox_list2 = [item["bbox"][0] for item in data2]
        bbox_areas2 = [calculate_bbox_area(bbox) for bbox in bbox_list2]
        max_area_index2 = bbox_areas2.index(max(bbox_areas2))
        keypointslist2 = data2[max_area_index2]['keypoints']
        keypoints2 = torch.tensor(keypointslist2, dtype=torch.float32)
        #dataset
        iddata.append((file_path1, keypoints1, file_path2, keypoints2, label))
#print(iddata)

dataset = SiameseDataset(iddata, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


#VAL DATA

with open('../KABR/annotation/idval.csv', "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    
    # 逐行读取数据
    for row in csv_reader:
        # 获取图像路径和标签
        image1_path, image2_path, label = row
        #目录名
        file_name1 = image1_path.split('/')[0]
        file_name2 = image2_path.split('/')[0]
        #总路径
        file_path1 = os.path.join(input_file_dir, image1_path)
        file_path2 = os.path.join(input_file_dir, image2_path)
    # save keypoints as json file  ZG0007.3/1.json
        if not os.path.exists(f"predictions/{image1_path}"):
            result_generator1 = model_pose(file_path1, pred_out_dir=f"predictions/{file_name1}")
            result = next(result_generator1)
        if not os.path.exists(f"predictions/{image2_path}"):
            result_generator2 = model_pose(file_path2, pred_out_dir=f"predictions/{file_name2}")
            result = next(result_generator2)
        
 # 生成的json文件读取
        # 将文件路径拆分为目录和文件名
        directory, filename = os.path.split(image1_path)
        # 将文件名的扩展名更改为 .json
        new_filename = os.path.splitext(filename)[0] + ".json"
        # 构造新的文件路径
        new_path1 = os.path.join(directory, new_filename)

        directory, filename = os.path.split(image2_path)

        new_filename = os.path.splitext(filename)[0] + ".json"

        new_path2 = os.path.join(directory, new_filename)

        with open(f"predictions/{new_path1}", "r") as json_file1:
            data1 = json.load(json_file1)

        bbox_list1 = [item["bbox"][0] for item in data1]
        bbox_areas1 = [calculate_bbox_area(bbox) for bbox in bbox_list1]
        max_area_index1 = bbox_areas1.index(max(bbox_areas1))
        keypointslist1 = data1[max_area_index1]['keypoints']

        keypoints1 = torch.tensor(keypointslist1, dtype=torch.float32)


        with open(f"predictions/{new_path2}", "r") as json_file2:
            data2 = json.load(json_file2)

        bbox_list2 = [item["bbox"][0] for item in data2]
        bbox_areas2 = [calculate_bbox_area(bbox) for bbox in bbox_list2]
        max_area_index2 = bbox_areas2.index(max(bbox_areas2))
        keypointslist2 = data2[max_area_index2]['keypoints']
        keypoints2 = torch.tensor(keypointslist2, dtype=torch.float32)
        valdata.append((file_path1, keypoints1, file_path2, keypoints2, label))

valdataset = SiameseDataset(valdata, transform=transform)
valdataloader = DataLoader(valdataset, batch_size=batch_size, shuffle=False)


with open('../KABR/annotation/idtest.csv', "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    
    # 逐行读取数据
    for row in csv_reader:
        # 获取图像路径和标签
        image1_path, image2_path, label = row
        #目录名
        file_name1 = image1_path.split('/')[0]
        file_name2 = image2_path.split('/')[0]
        #总路径
        file_path1 = os.path.join(input_file_dir, image1_path)
        file_path2 = os.path.join(input_file_dir, image2_path)
    # save keypoints as json file  ZG0007.3/1.json
        '''
        

        if not os.path.exists(f"predictions/{image1_path}"):
            result_generator1 = model_pose(file_path1, pred_out_dir=f"predictions/{file_name1}")
            result = next(result_generator1)
        if not os.path.exists(f"predictions/{image2_path}"):
            result_generator2 = model_pose(file_path2, pred_out_dir=f"predictions/{file_name2}")
            result = next(result_generator2)'''
        
 # 生成的json文件读取
        # 将文件路径拆分为目录和文件名
        directory, filename = os.path.split(image1_path)
        # 将文件名的扩展名更改为 .json
        new_filename = os.path.splitext(filename)[0] + ".json"
        # 构造新的文件路径
        new_path1 = os.path.join(directory, new_filename)

        directory, filename = os.path.split(image2_path)

        new_filename = os.path.splitext(filename)[0] + ".json"

        new_path2 = os.path.join(directory, new_filename)

        with open(f"predictions/{new_path1}", "r") as json_file1:
            data1 = json.load(json_file1)

        bbox_list1 = [item["bbox"][0] for item in data1]
        bbox_areas1 = [calculate_bbox_area(bbox) for bbox in bbox_list1]
        max_area_index1 = bbox_areas1.index(max(bbox_areas1))
        keypointslist1 = data1[max_area_index1]['keypoints']

        keypoints1 = torch.tensor(keypointslist1, dtype=torch.float32)


        with open(f"predictions/{new_path2}", "r") as json_file2:
            data2 = json.load(json_file2)

        bbox_list2 = [item["bbox"][0] for item in data2]
        bbox_areas2 = [calculate_bbox_area(bbox) for bbox in bbox_list2]
        max_area_index2 = bbox_areas2.index(max(bbox_areas2))
        keypointslist2 = data2[max_area_index2]['keypoints']
        keypoints2 = torch.tensor(keypointslist2, dtype=torch.float32)
        testdata.append((file_path1, keypoints1, file_path2, keypoints2, label))

testdataset = SiameseDataset(testdata, transform=transform)
testdataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)

# 定义孪生网络
'''
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(inplace=True)
        )
'''
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=4),
            nn.ReLU(inplace=True)
        )
        
        self.fc1 = nn.Linear(46208, 100)

        self.fc2 = nn.Linear(228, 100)
        self.fc3 = nn.Linear(100, 1)
        # 新增的全连接层处理关键点信息
        self.fc_keypoints = nn.Linear(34, 128)
        
    def forward_once(self, x, keypoints):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)

        keypoints_flat = keypoints.view(keypoints.size(0), -1)
        keypoints_embedding = self.fc_keypoints(keypoints_flat)

   #     print("Shape of x:", x.shape)
   #     print("Shape of keypoints_embedding:", keypoints_embedding.shape)
        
        # 将图像特征和关键点信息进行融合
        fused_features = torch.cat((x, keypoints_embedding), dim=1)
        
        return fused_features

    def forward(self, input1, keypoints1, input2, keypoints2):
        output1 = self.forward_once(input1, keypoints1)
        output2 = self.forward_once(input2, keypoints2)
        abs_diff = torch.abs(output1 - output2)
        final_output = self.fc2(abs_diff)
        final_output = self.fc3(final_output)
        # Apply sigmoid activation function
        final_output = torch.sigmoid(final_output)
        
        return final_output

# 创建孪生网络模型
model = SiameseNetwork()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
model.to(device)

# 设置损失函数和优化器
criterion = nn.BCELoss()
'''
1.optimizer SGD ADAM 2.L2 Regularization 3.Early Stopping
1. learning rate
'''
#optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.SGD(model.parameters(), lr=0.001)
#optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.15) 
optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.01)
# 训练模型
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for i, data in enumerate(dataloader):
        img1, keypoints1, img2, keypoints2, label = data
        img1, keypoints1, img2, keypoints2, label = img1.to(device), keypoints1.to(device), img2.to(device), keypoints2.to(device), label.to(device)

        # 清除之前的梯度
        optimizer.zero_grad()
        
        # 模型前向传播
        output = model(img1, keypoints1, img2, keypoints2)
        
        # 计算损失
        loss = criterion(output, label)
      #  print(loss)
        total_loss += loss.item()
        
        # 反向传播及参数更新
        loss.backward()
        optimizer.step()
    average_loss = total_loss / len(dataloader)
    writer.add_scalar('Training Loss', average_loss, epoch)

    model.eval()  # 设置模型为评估模式

    total_val_correct = 0
    with torch.no_grad():
        for i, data in enumerate(valdataloader):
            img1, keypoints1, img2, keypoints2, label = data
            img1, keypoints1, img2, keypoints2, label = img1.to(device), keypoints1.to(device), img2.to(device), keypoints2.to(device), label.to(device)
            outputs = model(img1, keypoints1, img2, keypoints2)
            reshape_label = label.squeeze().int()
        # Convert outputs to binary predictions (0 or 1)
        predictions = (outputs >= 0.5).squeeze().int()
        for i in range(predictions.size(0)):
            if predictions[i]==reshape_label[i] :
                total_val_correct = total_val_correct +1
    valaccuracy = total_val_correct / label.size(0)
    writer.add_scalar('Validation Accuracy', valaccuracy, epoch)
    #test
    #ZG0674 - ZG0742
    total_correct = 0

    model.eval()
    with torch.no_grad():
        for img1, keypoints1, img2, keypoints2, labels in testdataloader:
            # 将数据移动到设备上
            img1, keypoints1, img2, keypoints2, labels = img1.to(device), keypoints1.to(device), img2.to(device), keypoints2.to(device), labels.to(device)
            
            # 执行模型的前向传播
            outputs = model(img1, keypoints1, img2, keypoints2)
            reshape_label = labels.squeeze().int()
            # Convert outputs to binary predictions (0 or 1)
            predictions = (outputs >= 0.5).squeeze().int()

            for i in range(predictions.size(0)):
                if predictions[i]==reshape_label[i] :
                    total_correct = total_correct +1

    accuracy = total_correct / labels.size(0)
    writer.add_scalar('Test Accuracy', accuracy, epoch)
    print(reshape_label)
    print(predictions)    

    print('Epoch [{}/{}], Average Loss: {:.4f}, Validation Accuracy: {:.4f}, Test Accuracy: {:.4f}'.format(epoch+1, num_epochs, average_loss, valaccuracy, accuracy))


    # Early Stopping检查
    if valaccuracy > best_val_acc:
        best_val_acc = valaccuracy
        early_stopping_counter = 0  # 重置early_stopping_counter
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered. Training stopped.")
            break

print("Training Finished!")



