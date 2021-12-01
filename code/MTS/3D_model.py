from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import SGD,Adam

import pandas as pd
import os
import numpy as np
import glob
import keras
from keras.models import load_model

def c3d_model():
    input_shape = (23, 23, 2, 1)  # (image.raw, image.col, clip_num, channel)
    weight_decay = 0.005
    nb_classes = 15

    inputs = Input(input_shape)
    x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPool3D((2,2,1),strides=(2,2,1),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes,kernel_regularizer=l2(weight_decay))(x)
    x = Activation('softmax')(x)
    model = Model(inputs, x)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model


def wff_3D_model():
    input_shape = (23, 23, 2, 1)  # (image.raw, image.col, clip_num, channel)
    nb_classes = 15

    inputs = Input(input_shape)
    x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu')(inputs)
    x = MaxPool3D((2,2,1),strides=(2,2,1),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu')(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu')(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)
    # # #
    x = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu')(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu')(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation='softmax')(x)
    # x = Activation('softmax')(x)
    model = Model(inputs, x)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def wff_3D_modelv2():
    input_shape = (23, 23, 2, 1)  # (image.raw, image.col, clip_num, channel)
    nb_classes = 15
    weight_decay = 0.005

    inputs = Input(input_shape)
    x = Conv3D(64,(2,2,2),strides=(1,1,1),padding='same',
               activation='relu')(inputs)
    x = MaxPool3D((2,2,2),strides=(2,2,1),padding='same')(x)

    x = Conv3D(128,(2,2,2),strides=(1,1,1),padding='same',
               activation='relu')(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(128,(2,2,2),strides=(1,1,1),padding='same',
               activation='relu')(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)
    #
    x = Conv3D(256,(2,2,2),strides=(1,1,1),padding='same',
               activation='relu')(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256, (2, 2, 2), strides=(1, 1, 1), padding='same',
               activation='relu')(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(4096,activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(2048,activation='relu')(x)
    x = Dense(2048, activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation='softmax')(x)
    # x = Activation('softmax')(x)
    model = Model(inputs, x)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model


#  data loading
path = "MTS/Libras/"
train_data = []
train_label = []
test_data = []
test_label = []
for home, dirs, files in os.walk(path):
    for dir_name_temp in dirs:  # 遍历所有的文件夹
        dir_name = os.path.join(path, dir_name_temp)
        print('目前在文件夹' + dir_name_temp + '下面--------------------' + dir_name_temp)
        excel_Paths = []
        excel_Paths = glob.glob(os.path.join(dir_name, '*.xlsx'))
        excel_Paths.sort()
        # print(imagePaths)
        for excelPaths_item in excel_Paths:
            print(excelPaths_item)
            label = excelPaths_item.split('_')[1].split('.')[0]
            print('label',label)
            data_z = pd.read_excel(excelPaths_item, header=None)
            # print(data_z)
            # 46 rows x 23 columns
            # num1:0-22raw,
            # num2:23-45raw.
            data_z = np.asarray(data_z)
            print(data_z.shape[0], data_z.shape[1])  # 46*23
            # print(data_z)
            data_z = data_z.reshape((23, 23, 2, 1))
            print(data_z.shape)  # (2, 23, 23) to (23, 23, 2, 1)
            # print(data_z)

            label = int(label)-1
            if (dir_name_temp == 'test'):
                test_data.append(data_z)
                test_label.append(label)
            elif (dir_name_temp == 'train'):
                train_data.append(data_z)
                train_label.append(label)

train_data = np.array(train_data, dtype="float")
test_data = np.array(test_data, dtype="float")

from sklearn.model_selection import train_test_split  # 随机划分样本数据为训练集和测试集
x_train, x_val, y_train, y_val = train_test_split(
    train_data, train_label,
    test_size=0.2,
    # random_state=20,
    shuffle=True,
    stratify=train_label
)
# x_train = train_data
# y_train = train_label
# x_val = test_data
# y_val = test_label
print(x_train.shape)

from keras.utils import to_categorical
y_train_onehot = to_categorical(y_train)
y_val_onehot = to_categorical(y_val)
y_test_onehot = to_categorical(test_label)
print('y_train_onehot', y_train_onehot.shape)
print('y_val_onehot', y_val_onehot.shape)
print('y_test_onehot', y_test_onehot.shape)


# c3d = c3d_model()
# print(c3d.summary())
# model_h5 = c3d_model()
model = wff_3D_model()
# 2DConv input=(image.raw, image.col, image.channel),3D,but train.shpe=(num, image.raw, image.col, image.channel)
# 3DConv  input_shape=4D,,rain.shape=5D
lr = 0.005
sgd = SGD(lr=lr, momentum=0.9, nesterov=True)


for i in range(5):
    model = wff_3D_model()
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='model_h5/C3D_MTS_5serise_Libras_maxfbl_'+str(i)+'.h5',
            monitor='val_accuracy',  # 监控指标是精度。如果精度不再变大，则不更新权重，如果精度变大则更新权重
            save_best_only=True,  # 只保存最好的。
        )
    ]

    history = model.fit(x_train, y_train_onehot,
                        batch_size=16,  # 每批大小
                        epochs=200,
                        # validation_split=0.3,
                        callbacks=callbacks_list,
                        validation_data=(x_val, y_val_onehot),
                        verbose=2)

    # 曲线图
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
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'bo', label='acc_train')
    plt.plot(epochs, val_acc, 'b', label='acc_val')
    plt.title('acc of data')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.show()
    test_loss, test_acc = model.evaluate(test_data, y_test_onehot)
    print('test_loss, test_acc', test_loss, test_acc)

    model_l = load_model('model_h5/C3D_MTS_Libras_maxfbl_'+str(i)+'.h5')
    test_loss, test_acc = model_l.evaluate(test_data, y_test_onehot)
    print('test_loss, test_acc', test_loss, test_acc)


