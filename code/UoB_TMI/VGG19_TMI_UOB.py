'''keras下的vgg16，所有的层都训练，全部解冻 '''
import cv2
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras import optimizers
#指定可用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#网络模型-------
# 定义vgg16
import keras
from keras import models
from keras import layers
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import InceptionV3
from tensorflow.keras.backend import clear_session
from keras.models import load_model

conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(256, 255, 3))  # input_shape 是输入到网络中的图像张量的形状。这个参数完全是可选的，如果不传入这个参数，那么网络能够处理任意形状的输入。
print(conv_base.summary())
def build_model():
    model = models.Sequential()
    model.add(conv_base)  # 添加vgg16网络
    model.add(layers.Flatten())
    # original networks
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    print(model.summary())
    conv_base.trainable = True  # 冻结权重
    # 设置编译参数
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])  # 二分类，二元交叉熵
    return model


# 读取train数据
data_image = []
data_label = []
print(type(data_image))
root_dir = os.path.dirname(os.path.abspath('.'))  # 获取当前文件(这里是指\vgg16_python.py)所在目录的父目录的绝对路径,也就是项目所在路径psr_py
Path_img = root_dir + '/seizure_data_all_img/data_img/seizure_img_keras/train/'
print(Path_img)
output_re = open('result_vgg19.txt', 'a')
output_re.write(Path_img)

import glob
imagePaths = []
imagePaths = glob.glob(os.path.join(Path_img, '*.png'))
imagePaths.sort()
# print(imagePaths)
for imagePath in imagePaths:
    print(imagePath)
    image = cv2.imread(imagePath, 3)
    data_image.append(image)
    label_FNOSZ = str_list=imagePath.split(".", 3)[0][-4]# 拿到[ ]里面的标签
    print(label_FNOSZ)
    if (label_FNOSZ == 'S'):
        data_label.append('1')
    if(label_FNOSZ == 'Z'):
        data_label.append('2')
    if (label_FNOSZ == 'F'):
        data_label.append('3')
    if (label_FNOSZ == 'O'):
        data_label.append('4')
    if (label_FNOSZ == 'N'):
        data_label.append('5')

# 对图像数据做scale操作,
data_image = np.array(data_image, dtype="float") / 255.0
print(type(data_image))  # 效果：data.shape=[width,height,channels]
print(data_image.shape)  # <class 'numpy.ndarray'> (4000, 256, 255, 3)
data_label = np.array(data_label)

# test_--------------------test_data ------------------------------------------------------
root_dir = os.path.dirname(os.path.abspath('.'))  # 获取当前文件(这里是指\vgg16_python.py)所在目录的父目录的绝对路径,也就是项目所在路径psr_py
Path_img_test = root_dir + '/seizure_data_all_img/data_img/seizure_img_keras/test/'
print(Path_img_test)
import glob
imagePaths_test = []
imagePaths_test = glob.glob(os.path.join(Path_img_test, '*.png'))
imagePaths_test.sort()
# print(imagePaths)
# 读取数据
test_image = []
test_label = []

for imagePath in imagePaths_test:
    print(imagePath)
    image = cv2.imread(imagePath,3)
    test_image.append(image)
    label_FNOSZ = str_list=imagePath.split(".",3)[0][-4]# 拿到[ ]里面的标签
    print(label_FNOSZ)
    if (label_FNOSZ == 'S'):
        test_label.append(1)
    else:
        test_label.append(0)

# 对test图像数据做scale操作
test_image = np.array(test_image, dtype="float") / 255.0
print(type(test_image))  # 效果：data.shape=[width,height,channels]
print(test_image.shape)  # <class 'opnumpy.ndarray'> (4000, 256, 255, 3)
test_label = np.array(test_label)

# train随机划分样本数据为训练集和测试集
from sklearn.model_selection import train_test_split  # train随机划分样本数据为训练集和测试集
img_train, img_val, label_train, label_val = train_test_split(
    data_image, data_label,
    test_size=0.2,
    random_state=20,
    shuffle=True
)
print(label_val)
label_train[label_train != '1'] = '0'
label_val[label_val != '1'] = '0'
print(label_val)  # 按照五份进行分层抽样，后面变成二分类的标签
acc_all=0
sen_all=0
spec_all=0
auc_all=0
for i in range(3, 6):
    print('---------------------------------------'+ str(i) +"---------------------")
    import time
    start = time.process_time()
    # num_epochs = 40
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='TMI_UOB_vgg19_epoch200_m255_t1_'+str(i)+'.h5',
            monitor='val_acc',  # 监控指标是精度。如果精度不再变大，则不更新权重，如果精度变大则更新权重
            save_best_only=True,  # 只保存最好的。
        )
    ]
    model = build_model()  # 创建网络
    history = model.fit(img_train, label_train,
                        batch_size=16,  # 每批大小
                        epochs=200,
                        callbacks=callbacks_list,
                        validation_data=(img_val, label_val),
                        verbose=2
                        )
    print('fit_time=', time.process_time() - start)

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

    # test
    import os, cv2
    import numpy as np

    clear_session()


    model = load_model('TMI_UOB_vgg19_epoch200_m255_t1_'+str(i)+'.h5')
    print(model.summary())


    # pre = model_h5.predict(test_image)
    # print(pre)
    # 画出ROC
    y_pre_quant = model.predict_proba(test_image)  # [:,1]
    # print('y_pre_quant:', y_pre_quant)

    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt  # 绘制img

    fpr, tpr, thresholds = roc_curve(test_label, y_pre_quant)
    # print(fpr)
    # print(tpr)
    # print(thresholds)
    plt.plot(fpr, tpr, c="b", clip_on=False)  # clip_on 设为false便可以覆盖轴，不被轴挡着
    # plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], ls='--', c=".3")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve')
    plt.xlabel('False Positivite Rate (1 - specificity)')
    plt.ylabel('True Positivite Rate (sensitivite)')
    plt.grid(True)

    plt.show()
    AUC = auc(fpr, tpr)

    #  计算特异性和灵敏度
    y_pre_quant[y_pre_quant >= 0.5] = 1
    y_pre_quant[y_pre_quant <= 0.5] = 0
    # print(y_pre_quant)
    # print(test_label)

    true_positive = 0  # TP
    false_positive = 0  # FP
    false_negative = 0  # FN
    true_negative = 0  # TN
    for i in range(len(y_pre_quant)):
        if (y_pre_quant[i] == 1 and test_label[i] == 1):
            true_positive = true_positive + 1  #
        elif (y_pre_quant[i] == 0 and test_label[i] == 0):
            true_negative = true_negative + 1  #
        elif (y_pre_quant[i] == 1 and test_label[i] == 0):
            false_positive = false_positive + 1
        elif (y_pre_quant[i] == 0 and test_label[i] == 1):
            false_negative = false_negative + 1
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (false_positive + true_negative)

    print("sensitivity:", sensitivity)
    print("specificity:", specificity)
    print('AUC=', AUC)
    score = model.evaluate(test_image, test_label)
    acc_all = acc_all+score[1]
    sen_all = sen_all+sensitivity
    spec_all = spec_all+specificity
    auc_all = auc_all+AUC

    print(score)
    output_re.write(str(i)+'\n'+'loss+test_acc: ' + str(score) + '\n')
    output_re.write('sensitivity+specificity: ' + str(sensitivity) + '  \t' + str(specificity) + '\n')
    output_re.write('AUC: ' + str(AUC) + '\n')
    # output_re.close()

print("acc_avg" + str(acc_all/5))
print("sens_avg" + str(sen_all/5))
print("spec_avg" + str(spec_all/5))
print("auc_avg" + str(auc_all/5))

output_re.write("acc_avg" + str(acc_all/5) + '\n')
output_re.write("sens_avg" + str(sen_all/5) + '\n')
output_re.write("spec_avg" + str(spec_all/5) + '\n')
output_re.write("auc_avg" + str(auc_all/5) + '\n')
output_re.close()
#
