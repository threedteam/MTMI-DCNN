'''Adiac'''

import os, glob, cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras
from keras import models
from keras import layers, optimizers, regularizers
from keras.applications import VGG16
from keras.models import load_model
import numpy as np
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(33, 33, 3))  # input_shape 是输入到网络中的图像张量的形状。这个参数完全是可选的，如果不传入这个参数，那么网络能够处理任意形状的输入。
print(conv_base.summary())
def build_model():
    model = models.Sequential()
    model.add(conv_base)  # 添加vgg16网络
    model.add(layers.Flatten())
    # model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))  #
    model.add(layers.Dense(2048, activation='relu'))
    # model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.BatchNormalization())  #
    # model.add(layers.Dropout(0.1))
    model.add(layers.Dense(2, activation='softmax'))
    print(model.summary())
    conv_base.trainable = True  # 冻结权重
    # 设置编译参数
    model.compile(loss='categorical_crossentropy',  # binary_crossentropy --  categorical_crossentropy
                  optimizer=optimizers.Adam(lr=2e-5),  # lr=2e-5
                  metrics=['acc'])  # 二分类，二元交叉熵
    return model

## 读取数据
train_data = []
train_label = []
test_data = []
test_label = []
root_dir = os.path.dirname(os.path.abspath('.'))  # 获取当前文件(这里是指\vgg16_python.py)所在目录的父目录的绝对路径,也就是项目所在路径psr_py
Path_img = root_dir + '/UCR_other/UCR_TMI/202108_new/SonyAIBORobotSurface2/'
output_re = open('result_TMI202108.txt', 'a')
output_re.write(Path_img)
print(Path_img)
for home, dirs, files in os.walk(Path_img):
    for dir_name_temp in dirs:  # 遍历所有的文件夹
        dir_name = os.path.join(Path_img, dir_name_temp)
        print('目前在文件夹' + dir_name_temp + '下面--------------------' + dir_name_temp)
        imagePaths = []
        imagePaths = glob.glob(os.path.join(dir_name, '*.png'))
        imagePaths.sort()
        # print(imagePaths)
        for imagePath in imagePaths:
            print(imagePath)
            image = cv2.imread(imagePath, 3)
            # image_resize = cv2.resize(image, (200, 200))
            label_FNOSZ = str_list = imagePath.split(".")[-2].split('[')[1].split(']')[0]  # 拿到[ ]里面的标签
            print(type(label_FNOSZ))
            print(label_FNOSZ)
            print(label_FNOSZ)
            if(label_FNOSZ=='1'):
                label=0
            else:
                label=1

            if(dir_name_temp == 'test'):
                test_data.append(image)
                test_label.append(label)

            elif(dir_name_temp == 'train'):
                train_data.append(image)
                train_label.append(label)

# 对图像数据做scale操作,
train_data = np.array(train_data, dtype="float") / 255.0
print(type(train_data))  # 效果：data.shape=[width,height,channels]
print(train_data.shape)  # <class 'numpy.ndarray'> (4000, 256, 255, 3)
train_label = np.array(train_label)
test_data = np.array(test_data, dtype="float") / 255.0
test_label = np.array(test_label)

from sklearn.model_selection import train_test_split  # 随机划分样本数据为训练集和测试集
x_train, x_val, y_train, y_val = train_test_split(
    train_data, train_label,
    test_size=0.2,
    random_state=20,
    shuffle=True,
    stratify=train_label
)

from keras.utils import to_categorical
y_train_onehot = to_categorical(y_train)
y_val_onehot = to_categorical(y_val)
y_test_onehot = to_categorical(test_label)
print('y_train_onehot', y_train_onehot.shape)
print('y_val_onehot', y_val_onehot.shape)
print('y_test_onehot', y_test_onehot.shape)

print(y_train, y_val)
acc_all = 0
loss_all = 0

for i in range(5):
    model = build_model()  # 创建网络
    print(model.summary())
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='../model_h5/TMI_VGG16_bestmodel_SonyAIBORobotSurface2_maxfbl_'+str(i)+'.h5',
            monitor='val_acc',  # 监控指标是精度。如果精度不再变大，则不更新权重，如果精度变大则更新权重
            save_best_only=True,  # 只保存最好的。
        )
    ]
    history = model.fit(x_train, y_train_onehot,
                        batch_size=8,  # 每批大小
                        epochs=100,
                        # validation_split=0.3,
                        callbacks=callbacks_list,
                        validation_data=(x_val, y_val_onehot),
                        verbose=2
                        )
    print(history.history.keys())
    # mae_history = history.history['val_loss']
    import matplotlib.pyplot as plt
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='loss_train')  # Training lossValidation loss  scatter plot
    plt.plot(epochs, val_loss, 'b', label='loss_val')
    plt.title('loss of data')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    plt.savefig('loss_keras00.png')
    plt.clf()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.plot(epochs, acc, 'bo', label='acc_train')
    plt.plot(epochs, val_acc, 'b', label='acc_val')
    plt.title('acc of data')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.show()
    model = load_model('../model_h5/TMI_VGG16_bestmodel_SonyAIBORobotSurface2_maxfbl_'+str(i)+'.h5')
    # 画出ROC
    y_pre_quant = model.predict(test_data)  # [:,1]
    print('y_pre_quant:', y_pre_quant)
    test_loss, test_acc = model.evaluate(test_data, y_test_onehot)
    acc_all = acc_all+test_acc
    loss_all = loss_all+test_loss
    print('test: ', test_loss, test_acc)
    output_re.write('test_acc:'+str(test_acc) + '\n')

print('loss_avg,acc_avg:', loss_all/5, acc_all/5)
# print('AUC_all,sens_all, spes_all', AUC_all, sens_all/5, spes_all/5)
output_re.write('acc_avg:'+str(acc_all / 5)+'\n')
output_re.close()


