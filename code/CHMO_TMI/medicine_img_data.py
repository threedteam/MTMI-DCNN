''' 处理中药数据 ，筛选数据+滤波'''
import pandas as pd
import numpy as np
import os, glob, math
root_dir = os.path.dirname(os.path.abspath('..'))  # 获取当前文件(这里是指\vgg16_python.py)所在目录的父目录的绝对路径,也就是项目所在路径psr_py
data_dir_A = root_dir + '/sensor_data/Chinese_medicine/1'  # 根据项目所在路径，找到用例所在的相对项目的路径
print(data_dir_A)
imagePaths = glob.glob(os.path.join(data_dir_A, '*.txt'))# 获取A_Z下面所有的文件
imagePaths.sort()
for imagePath in imagePaths:  # 遍历所有的文件，进行操作,共100个文件
    print(imagePath)
    txt_label1 = imagePath.split('-')[-2]
    txt_label2 = imagePath.split('-')[1]
    print(txt_label2)
    print(txt_label1)
    medcine_data = pd.read_csv(imagePath, delimiter=' ', header=None)
    # print(medcine_data)  # 传感器数据，txt中前面时信息，0-4时信息，5-10：温湿度压强；11-28：气体传感器数据，11号传感器数据时坏的。所以我们取12-28列.
    # 选择一个传感器数据进行处理；14号
    # ['t_idx', 'bike_in_cnt']是取特定的列
    # df1['bike_in_cnt'] > 10是取特定的行
    # medcine_data = medcine_data[[14]][medcine_data[4] == 'P15']  # p15 data
    medcine_data = medcine_data[[14]]  # all data
    print(len(medcine_data))
    print(medcine_data.iloc[len(medcine_data) - 1])
    # for i in range(3):
    #     if (len(medcine_data) >= 360):
    #         break
    #     new = pd.DataFrame({14: medcine_data.iloc[len(medcine_data) - 1]})
    #     medcine_data = medcine_data.append(new, ignore_index=True)  # ignore_index=True,表示不按原来的索引，从0开始自动递增

    print(medcine_data)
    print(medcine_data.shape)
    medcine_data = medcine_data.iloc[0:360, :]  # 选择进样阶段的数据  [364 rows x 1 columns]
    print(medcine_data)
    data_list = np.asarray(medcine_data).reshape((1, 360))
    print(data_list.shape)
    print(data_list)  # 每个txt数据都被分割成2*180的
    data_psr = {}
    for i in range(0, len(data_list)):  # 本来有58个样本，现在删除空气和混合物的，就只有30个
        # X = sensor_data_clean.iloc[i][::10]
        X = data_list[i]
        print(X.shape)  # (,1)
        N = len(X)
        m = math.ceil((N / 2))
        # m = 45
        tau = 1
        L = N - (m - 1) * tau
        data_psr_temp = pd.DataFrame(np.arange((L * m)).reshape(L, m))
        print(data_psr_temp.shape)
        for k in range(0, L):
            for j in range(0, m):
                data_psr_temp.iloc[k, j] = X[((j * tau) + k)]  # 循环结束生成一个矩阵,循环覆盖
        data_psr[i] = data_psr_temp  # 每一行对应一个矩阵对应一个样本
        # 把样本矩阵转化成灰度图保存到文件夹
        array = np.array(data_psr[i])
        xmax = max(map(max, array))
        xmin = min(map(min, array))
        # 把矩阵统一映射到0-255
        for width in range(array.shape[0]):
            for hight in range(array.shape[1]):
                array[width][hight] = round((255 * (array[width][hight] - xmin) / (xmax - xmin)))
        print(array)
        from PIL import Image
        im = Image.fromarray(np.uint8(array))  # 把array转化成image
        im = im.convert('RGB')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’  0是黑,255是白,所以猜测右上角一点是黑的,其余的是白色的
        print('medicine_img_m180_t1_N360/' + txt_label2 + '.' + txt_label1 + '.' + str(i) + '.png')
        im.save('medicine_img_m180_t1_N360/' + txt_label2 + '.' + txt_label1 + '.' + str(i) + '.png')