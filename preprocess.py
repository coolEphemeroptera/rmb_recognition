import os
import csv
import cv2
import numpy as np
import random
import shutil
import tfr_tools as tfr

label_dict = {' 0.1':0,' 0.2':1,' 0.5':2,' 1':3,' 2':4,' 5':5,' 10':6,' 50':7,' 100':8}

def show(img):
    cv2.namedWindow(' ',cv2.WINDOW_NORMAL)
    cv2.imshow(' ',img)
    cv2.waitKey(0)

def Reading_Label_CSV(csv_path,id):
    csv_file = csv.reader(open(csv_path, 'r'))
    for line in csv_file:
        if id==line[0]:
            return line[1]

def onehot(label):
    x = np.zeros(shape=9)
    x[label] = 1
    return x

def saving_tfr(file_list,data_dir,tfr_path,N=10):
    # 获取N组区间下标
    data_total = len(file_list)
    index = [int(i) for i in np.linspace(0, data_total, N+1)]
    # 遍历每组
    for i in range(N):
        datas = []
        labels = []
        idxs = np.arange(index[i],index[i+1]-1)
        for filename in file_list[index[i]:index[i+1]]:
            # 读取图片
            img = cv2.imread(os.path.join(data_dir,filename))
            # resize
            img = cv2.resize(img,(256,128))
            # show(img)
            # 读取标签并转化为onehot
            label_ley = Reading_Label_CSV(csv_path, filename)
            label = label_dict[label_ley]
            label = onehot(label)
            # 记录
            datas.append(img)
            labels.append(label)
            print('成功处理第%d组图片：%s..' % (i + 1, filename))
        # 保存tfr
        tfr.Saving_Batch_TFR(tfr_path, idxs,datas, labels, i, N-1)

def copy_into_other_dir(src_dir,dst_dir,file_list):
    # 清除目标文件夹
    dst_list = os.listdir(dst_dir)
    for dst_file in dst_list:
        os.remove(os.path.join(dst_dir,dst_file))
        print('正在删除文件%s..'%dst_file)
    # 拷贝
    for file in file_list:
        shutil.copy(os.path.join(src_dir,file),os.path.join(dst_dir,file))
        print('正在拷贝文件%s'%file)


if __name__ == '__main__':

    # 设置数据路径
    data_dir = r'../rmb_data/train_data'
    test_dir = r'../rmb_data/test_data'
    csv_path = r'../rmb_data/train_face_value_label.csv'

    # 数据集列表
    data_list = os.listdir(data_dir)
    data_total = len(data_list)

    # 随机抽取70%作为训练集，30%作为测试集
    train_total = int(0.7 * data_total)  # 训练集总量
    test_total = data_total - train_total  # 测试集总量
    train_list = random.sample(data_list, train_total)  # 训练样本列表
    test_list = [item for item in data_list if item not in train_list]  # 测试样本列表

    # 保存训练集
    saving_tfr(train_list,data_dir,r'./train_tfr/rmb',10)

    # 拷贝测试图片
    copy_into_other_dir(data_dir,test_dir,test_list)
