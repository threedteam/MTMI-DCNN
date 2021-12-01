'''
训练中药数据集
先划分测试集和训练集,然后,把训练集划出来一部分验证集合
 '''
import cv2
import os
import numpy as np
import glob
from sklearn.model_selection import StratifiedKFold
from keras import optimizers
from keras import Model
from keras import backend as K
from keras.models import load_model
from keras.utils.np_utils import to_categorical  # one-hot
print(K.tensorflow_backend._get_available_gpus())  # 查看keras认可的GPU
#指定可用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 查看正在使用的GPU
import tensorflow as tf
print('tf.__version__', tf.__version__)
if tf.test.gpu_device_name():
    print('Default GPU Device 正在使用的: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

#网络模型-------
# 定义vgg16
import keras
from keras import models
from keras import layers
from keras.applications import InceptionV3, VGG16
import time

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(181, 180, 3))  # input_shape 是输入到网络中的图像张量的形状。这个参数完全是可选的，如果不传入这个参数，那么网络能够处理任意形状的输入。
print(conv_base.summary())

conv_base.trainable = True  #
def build_model():
    # model_h5 = models.Sequential()
    # model_h5.add(conv_base)  # 添加vgg16网络
    # model_h5.add(layers.Flatten())
    # model_h5.add(layers.Dense(1024, activation='relu'))
    # model_h5.add(layers.Dense(1024, activation='relu'))
    # model_h5.add(layers.Dense(1, activation='sigmoid'))
    # print(model_h5.summary())
    ### 修改
    model1 = layers.Flatten()(conv_base.output)
    # model2 = layers.Dense(2048, activation='relu', name='fc1')(model1)
    model3 = layers.Dense(4096, activation='relu', name='fc1_1')(model1)
    # model_BN = layers.BatchNormalization()(model3)  # 小数据样本考虑冻结批
    model4 = layers.Dense(2048, activation='relu', name='fc2')(model3)
    # model_drop = layers.Dropout(0.4)(model4) # 减少过拟合,dropout正则化技术
    model5 = layers.Dense(1024, activation='relu', name='fc3')(model4)
    # model_drop2 = layers.Dropout(0.1)(model5)  # 减少过拟合,dropout正则化技术
    model6 = layers.Dense(5, activation='softmax')(model5)  #
    model_vgg_new = Model(conv_base.input, model6, name='vgg16_keras')

    conv_base.trainable = True
    # set_trainable = False
    # for layer in conv_base.layers:
    #     layer.trainable = set_trainable
    # 设置编译参数
    model_vgg_new.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=2e-5),  # lr=1e-6,RMSprop
                  metrics=['acc'])

    # model_vgg_new.compile(loss='categorical_crossentropy',
    #               optimizer=optimizers.RMSprop(lr=2e-5),
    #               metrics=['acc'])  # 5分类，多元交叉熵
    return model_vgg_new

# 读取数据
train_data = []
train_label = []
# path_train = []
test_data = []
test_label = []

root_dir = os.path.dirname(os.path.abspath('.'))  # 获取当前文件(这里是指\vgg16_python.py)所在目录的父目录的绝对路径,也就是项目所在路径psr_py
Path_img = root_dir + '/Chinese_medicine/TMI_test_train/'
print(Path_img)
print("文件夹", Path_img)
output_re = open('result_medicine_202110.txt', 'a')
output_re.write(Path_img)
output_re.write("dataAll_VGG16" + "\t")

for home, dirs, files in os.walk(Path_img):
    for dir_name_temp in dirs:  # 遍历所有的文件夹
        dir_name = os.path.join(Path_img, dir_name_temp)
        print(dir_name_temp)
        imagePaths = glob.glob(os.path.join(dir_name + "/", '*.png'))
        imagePaths.sort()
        for imagePath in imagePaths:
            print(imagePath)
            image = cv2.imread(imagePath, 3)
            # image_resize = cv2.resize(image, (212, 212))

            print(image.shape)
            label_FNOSZ = str_list = imagePath.split("/")[-1].split('.')[0]  # 拿到[ ]里面的标签
            print(type(label_FNOSZ))
            print(label_FNOSZ)
            if label_FNOSZ == '7':
                label = 0
            elif label_FNOSZ == '36':
                label = 1
            elif label_FNOSZ == '47':
                label = 2
            elif label_FNOSZ == '48':
                label = 3
            elif label_FNOSZ == '80':
                label = 4
            if(dir_name_temp == 'train'):
                train_data.append(image)
                train_label.append(label)
            elif(dir_name_temp == 'test'):
                test_data.append(image)
                test_label.append(label)

# 对图像数据做scale操作,
train_data = np.array(train_data, dtype="float") / 255.0
train_label = np.array(train_label)
test_data = np.asarray(test_data, dtype="float") / 255.0
test_label = np.asarray(test_label)

print('train_label:', train_label)
print('test_label:', test_label)

# # 手动shuffle
# np.random.seed(200)
# np.random.shuffle(train_data)
# np.random.seed(200)
# np.random.shuffle(train_label)
#
# np.random.seed(200)
# np.random.shuffle(test_data)
# np.random.seed(200)
# np.random.shuffle(test_label)
#
# print('train_label:', train_label)
#
# print('train_label:', test_label)

# test_set_partition
from sklearn.model_selection import train_test_split  # 随机划分样本数据为训练集和测试集
x_train, x_val, y_train, y_val = train_test_split(
    train_data, train_label,
    test_size=0.1,
    random_state=20,
    shuffle=True,
    stratify=train_label  #
)

print('x_train, y_train:', x_train.shape, y_train.shape)
print('x_val, y_val:', x_val.shape, y_val.shape)
print('x_test, y_test', test_data.shape, test_label.shape, test_label)

print(y_train.shape, y_train)
print(y_val.shape, y_val)
label_train_one_hot = to_categorical(y_train)
label_val_one_hot = to_categorical(y_val)
label_test_one_hot = to_categorical(test_label)
print(label_train_one_hot.shape, label_val_one_hot.shape, label_test_one_hot.shape)

acc_all = 0
for i in range(5, 10):
    model = build_model()  # 创建网络
    print(model.summary())
    # conv_base.trainable = True  # 先解冻 conv_base，然后冻结其中的部分层。
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='../model_h5/202110_medicine_'+str(i)+'.h5',
            monitor='val_acc',  # 监控指标是精度。如果精度不再变大，则不更新权重，如果精度变大则更新权重
            save_best_only=True,  # 只保存最好的。
        )
    ]
    start = time.process_time()
    history = model.fit(x_train, label_train_one_hot,
                        batch_size=16,  # 每批大小
                        epochs=200,
                        # validation_split=0.3,
                        callbacks=callbacks_list,
                        validation_data=[x_val, label_val_one_hot],
                        verbose=2
                        )
    print('fit_time=', time.process_time() - start)
    output_re.write('fit_time=' + str(time.process_time() - start) + '\t')

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

    model.save('../model_h5/202110_medicine_endmodel'+str(i)+'.h5')
    output_re.write('../model_h5/202110_medicine_endmodel'+str(i)+'.h5' + '\n')
    model_end = load_model('../model_h5/202110_medicine_endmodel'+str(i)+'.h5')
    end_loss, end_acc = model_end.evaluate(test_data, label_test_one_hot)
    print('end_test:', end_loss, end_acc)
    output_re.write('end_test:' + str(end_loss) + '\t' + str(end_acc) + '\n')

    # ceshi
    model_best = load_model('../model_h5/202110_medicine_'+str(i)+'.h5')
    y_pre_quant = model_best.predict(test_data)  # [:,1]
    print('y_pre_quant:', y_pre_quant)
    loss, acc = model_best.evaluate(test_data, label_test_one_hot)
    print('test:', loss, acc)
    output_re.write('test:' + str(loss) + '\t' + str(acc) + '\n')
    # print(loss)
    acc_all = acc_all+acc

print('avg_acc5', acc_all/5)
output_re.write('test:'+str(acc_all / 5)+'\n')
output_re.close()
