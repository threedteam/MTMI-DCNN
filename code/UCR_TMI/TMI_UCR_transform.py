# 读取UOB数据
import os
import glob
import pandas as pd
import numpy as np
import math
from PIL import Image
from pyts.approximation import PiecewiseAggregateApproximation
data_root = os.path.dirname(os.path.abspath('..'))
data_dir = data_root + "/sensor_data/UCR_other/202108_new/"  #  data set path
print(data_dir)
for home, dirs, files in os.walk(data_dir):
    for dir_name_temp in dirs:  # 遍历所有的文件夹
        dir_name = os.path.join(data_dir, dir_name_temp)
        print('目前在文件夹' + dir_name_temp + '下面----------------------------------------' + dir_name_temp)
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        imagePaths = glob.glob(os.path.join(dir_name + "/", '*.tsv'))
        for (imagePath) in imagePaths:  # 遍历所有的文件，进行操作,共800个文件
            print(imagePath)
            dir_name_temp_inner = imagePath.split("/")[-1]  # 获取文件的标识
            print('train-test', dir_name_temp_inner)
            data_raw = pd.read_csv(imagePath, header=None, sep='\t')  # 读取到文件数据
            print('data_raw.shape', data_raw.shape, data_raw.shape[1])  # 28*287
            print(data_raw)
            # data_raw.drop(data_raw.columns[[0]], axis=1, inplace=True)
            print(data_raw)
            if ('TEST' in dir_name_temp_inner):
                test_data.append(data_raw.iloc[:, 1:data_raw.shape[1]])  # 1-
                test_label.append(data_raw.iloc[:, 0])
            elif ('TRAIN' in dir_name_temp_inner):
                train_data.append(data_raw.iloc[:, 1:data_raw.shape[1]])  # 1-286
                train_label.append(data_raw.iloc[:, 0])  # 0
                print(data_raw.iloc[:, 0])


        train_data = np.asarray(train_data)
        print(train_data.shape)
        train_data = train_data.reshape((-1, (train_data.shape[2]), 1))
        train_label = np.asarray(train_label)
        print(train_label.shape)
        train_label = train_label.reshape((train_label.shape[1], 1))
        test_data = np.asarray(test_data)
        test_data = test_data.reshape((-1, (test_data.shape[2]), 1))
        test_label = np.asarray(test_label)
        test_label = test_label.reshape((test_label.shape[1], 1))
        print(train_data.shape, test_data.shape)
        print(train_label.shape, test_label.shape)
        # print(train_label)
        print(len(train_data), len(train_label), str(train_label[0][0]))
        data_train_psr = {}
        data_test_psr = {}
        for train_data_i in range(len(train_data)):
            X = train_data[train_data_i]
            print(X.shape)  # (,1)  # (28, 143) 0  (143,)
            N = len(X)
            # m = math.ceil((N / 2))
            m = math.ceil((N / 2))
            tau = 1
            L = N - (m - 1) * tau
            data_psr_temp = pd.DataFrame(np.arange((L * m)).reshape(L, m))
            print(data_psr_temp.shape)
            for k in range(0, L):
                for j in range(0, m):
                    data_psr_temp.iloc[k, j] = X[((j * tau) + k)]  # 循环结束生成一个矩阵,循环覆盖
            data_train_psr[train_data_i] = data_psr_temp  # 每一行对应一个矩阵对应一个样本
            # 把样本矩阵转化成灰度图保存到文件夹
            array = np.array(data_train_psr[train_data_i])
            xmax = max(map(max, array))
            xmin = min(map(min, array))
            # 把矩阵统一映射到0-255
            for width in range(array.shape[0]):
                for hight in range(array.shape[1]):
                    array[width][hight] = round((255 * (array[width][hight] - xmin) / (xmax - xmin)))
            # print(array)
            from PIL import Image
            im = Image.fromarray(np.uint8(array))  # 把array转化成image
            im = im.convert('RGB')  # 这样才能转为灰度图
            print('UCR_TMI/'+dir_name_temp+'/train/' + str(train_data_i) + '.' + str(train_label[train_data_i]) + '.png')
            im.save('UCR_TMI/1num_2/'+dir_name_temp+'/train/' + str(train_data_i) + '.' + str(train_label[train_data_i]) + '.png')

        for test_data_i in range(len(test_data)):
            X = test_data[test_data_i]
            # print(X)
            print(X.shape)  # (,1)  # (28, 143) 0  (143,)
            N = len(X)
            # m = math.ceil((N / 2))
            m = math.ceil((N / 2))
            tau = 1
            L = N - (m - 1) * tau
            data_psr_temp = pd.DataFrame(np.arange((L * m)).reshape(L, m))
            print(data_psr_temp.shape)
            for k in range(0, L):
                for j in range(0, m):
                    data_psr_temp.iloc[k, j] = X[((j * tau) + k)]  # 循环结束生成一个矩阵,循环覆盖
            data_test_psr[test_data_i] = data_psr_temp  # 每一行对应一个矩阵对应一个样本
            # 把样本矩阵转化成灰度图保存到文件夹
            array = np.array(data_test_psr[test_data_i])
            xmax = max(map(max, array))
            xmin = min(map(min, array))
            # 把矩阵统一映射到0-255
            for width in range(array.shape[0]):
                for hight in range(array.shape[1]):
                    array[width][hight] = round((255 * (array[width][hight] - xmin) / (xmax - xmin)))
            # print(array)
            from PIL import Image
            im = Image.fromarray(np.uint8(array))  # 把array转化成image
            im = im.convert('RGB')  # 转为灰度图
            print('UCR_TMI/'+dir_name_temp+'/test/' + str(test_data_i) + '.' + str(test_label[test_data_i]) + '.png')
            im.save('UCR_TMI/1num_2/'+dir_name_temp+'/test/' + str(test_data_i) + '.' + str(test_label[test_data_i]) + '.png')
