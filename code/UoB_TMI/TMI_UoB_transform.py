# 癫痫数据预处理，提取切割255*256的尺寸
import os
import glob
import pandas as pd
import numpy as np
import math

'''定义一个重构的函数,传入一个二维的数据，转化成三维的，然后转化成图片'''
root_dir = os.path.dirname(os.path.abspath('.'))  # 获取当前文件(这里是指\vgg16_python.py)所在目录的父目录的绝对路径,也就是项目所在路径psr_py
data_dir_A = root_dir + '/sensor_data/Bonn_seizure_data/A_Z'  # 根据项目所在路径，找到用例所在的相对项目的路径
print(data_dir_A)
imagePaths = glob.glob(os.path.join(data_dir_A, '*.txt'))# 获取A_Z下面所有的文件
imagePaths.sort()
for imagePath in imagePaths: # 遍历所有的文件，进行操作
    print(imagePath)
    txt_label = imagePath.split("/")[-1][0:4]  # 获取文件的标识
    print(txt_label)
    data_raw=pd.read_table(imagePath,header=None).iloc[0:4080,:]
    print(data_raw)
    print(data_raw.shape)
    # 切分数据，4096/510=8
    data_list=np.asarray(data_raw).reshape((8, 510))
    print(data_list.shape)
    print(data_list)# 每个txt数据都被分割成8*510的
    # reconstruction(data_list,txt_label)
    data_psr = {}
    for i in range(0, len(data_list)):  # 本来有58个样本，现在删除空气和混合物的，就只有30个
        # X = sensor_data_clean.iloc[i][::10]
        X = data_list[i]
        print(X.shape)  # (,1)
        N = len(X)
        # m = math.ceil((N / 2))
        m = 32
        tau = 8
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
        im = im.convert('RGB')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
        print('data_img(m32_t8)/A_Z/' + txt_label + '.' + str(i) + '.png')
        im.save('data_img(m32_t8)/A_Z/' + txt_label + '.' + str(i) + '.png')


