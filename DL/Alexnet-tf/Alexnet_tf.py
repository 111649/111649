from tensorflow.keras import layers, models, Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt

#使用keras funciton api的方法搭建网络
def AlexNet(im_height=224, im_width=224, class_num=1000):#定义一个方法 传入：图像高度 图像宽度 分类类别
    # tensorflow中的tensor通道排序是NHWC
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")  # output(None, 224, 224, 3)
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)   #valid和same都不能满足输出，因此需要手动padding处理 output(None, 227, 227, 3)
    x = layers.Conv2D(48, kernel_size=11, strides=4, activation="relu")(x)       # output(None, 55, 55, 48)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x) #padding默认等于valid  # output(None, 27, 27, 48)
    x = layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(x)  # output(None, 27, 27, 128)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 13, 13, 128)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 128)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 6, 6, 128)

    x = layers.Flatten()(x)                         # output(None, 6*6*128=4608)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation="relu")(x)    # output(None, 2048)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation="relu")(x)    # output(None, 2048)
    x = layers.Dense(class_num)(x)                  # output(None, 5)
    predict = layers.Softmax()(x)#将输出转化成为一个概率分布

    model = models.Model(inputs=input_image, outputs=predict)
    return model


###################################################
########  train
###########################################
def train(image_path, im_height = 224, im_width = 224, batch_size = 3, epochs = 10):
    train_dir = image_path + "train"
    validation_dir = image_path + "val"

    # create direction for saving weights
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")#在指定的路径下创建文件夹 用来保存训练模型的权重

    #keras模块提供的图片生成器：可以载入文件夹下的图片生成器并对其进行预处理
    train_image_generator = ImageDataGenerator(rescale=1. / 255,#压缩像素同时进行随机水平反转
                                               horizontal_flip=True)
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)#定义验证集生成器
    #读取训练集图像文件
    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,#训练集目录
                                                               batch_size=batch_size,#每一批图像数据的数目
                                                               shuffle=True,#是否随机打乱
                                                               target_size=(im_height, im_width),#输入网络的尺寸大小
                                                               class_mode='categorical')#分类的方式
    total_train = train_data_gen.n #获得训练集训练样本的个数

    # get class dict
    class_indices = train_data_gen.class_indices#字典类型，返回每个类别和其索引
    print(class_indices)

    # 将key和value进行反转 得到反过来的字典 (目的：在预测的过程中通过索引直接对应到类别中)
    inverse_dict = dict((val, key) for key, val in class_indices.items())
    # python对象转换成json对象的一个过程，生成的是字符串。
    json_str = json.dumps(inverse_dict, indent=4)#将所得到的字典写入到json文件当中
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    #读取验证集图像文件
    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')
    total_val = val_data_gen.n

    model = AlexNet(im_height=im_height, im_width=im_width, class_num=3)#实例化网络
    model.summary()#可以看到模型的参数信息

    # using keras high level api for training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),#配置模型
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),# 有softmax处理处理用false 没有用true
                  metrics=["accuracy"])#所要监控的指标
    #定义回调函数(保存模型的一些规则)的列表 这里只用了一个回调函数：控制保存模型的参数
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlex.h5',#保存模型的位置:当前文件夹下
                                                    save_best_only=True,#是否保存最佳参数 还是保存最后的训练参数
                                                    save_weights_only=False,#是否只保存权重 如果不止权重文件还有模型文件，这样就不需要创建网络直接调用模型文件即可
                                                    monitor='val_loss')]#所监控的参数：验证集的损失 判断是不是最佳，变小的话模型效果就会变好

    # 训练过程的一些信息保存在history中
    history = model.fit(x=train_data_gen,#训练集生成器
                        steps_per_epoch=total_train // batch_size,#每一轮要迭代多少次即一个epoch要迭代多少次 //是除
                        epochs=epochs,#迭代多少轮
                        validation_data=val_data_gen,#给定验证集生成器
                        validation_steps=total_val // batch_size,#验证集的时候没有dropout fit方法自动实现了
                        callbacks=callbacks)#

    # plot loss and accuracy image
    history_dict = history.history#通过这样的方法可以获取到数据字典 保存了训练集的损失和准确率，验证集的损失和准确率
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["acc"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_acc"]

    # figure 1
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    # figure 2
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()

    history = model.fit_generator(generator=train_data_gen,
                                  steps_per_epoch=total_train // batch_size,
                                  epochs=epochs,
                                  validation_data=val_data_gen,
                                  validation_steps=total_val // batch_size,
                                  callbacks=callbacks)
    # model.save('save_weights/1/', save_format='tf')


#########################################################
## test
########################################################
def test(img_dir, im_height = 224, im_width = 224):
    # load image
    img = Image.open(img_dir)#上一层的目录中放了图片
    # resize image to 224x224
    img = img.resize((im_width, im_height))#对图像进行缩放
    plt.imshow(img)


    # scaling pixel value to (0-1)
    img = np.array(img) / 255.

    # Add the image to a batch where it's the only member. 扩充图片维度，输入到网络中必须是(batch 宽 高 深)
    img = (np.expand_dims(img, 0))

    # read class_indict
    try:
        json_file = open('./class_indices.json', 'r')#读取之前保存好的json文件
        class_indict = json.load(json_file)#对应的类别信息
    except Exception as e:
        print(e)
        exit(-1)

    model = AlexNet(class_num=3)#实例化模型
    model.load_weights("./save_weights/myAlex.h5")#载入模型
    result = np.squeeze(model.predict(img))#进行预测 得到的结果有batch维度，用squeeze压缩 得到概率分布
    predict_class = np.argmax(result)#获取概率最大的值所对应的索引
    print(class_indict[str(predict_class)], result[predict_class])#得到分类所属类别
    plt.show()

#########################################################
## convert  tflite_convert --output_file=[tflite文件生成的路径] --graph_def_file=[pb文件所在的路径] --input_arrays=[输入数组] --output_arrays=[输出数组]
########################################################
def create_graph(model_path):
    with tf.gfile.FastGFile(os.path.join(model_path), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def print_io_arrays(pb):
    gf = tf.GraphDef()
    m_file = open(pb, 'rb')
    gf.ParseFromString(m_file.read())

    with open('gfnode.txt', 'a') as the_file:
        for n in gf.node:
            the_file.write(n.name + '\n')

    file = open('gfnode.txt', 'r')
    data = file.readlines()
    print("output name = ")
    print(data[len(data) - 1])
    print("Input name = ")
    file.seek(0)
    print(file.readline())

#########################################################
## lite_test
########################################################
def lite_test(lite_model_file, img_path):
    interpreter = tf.lite.Interpreter(model_path=lite_model_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    floating_model = input_details[0]['dtype'] == np.float32
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(img_path).resize((width, height))
    if floating_model:
        img = np.float32(img) / 255
    input_data = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    # read class_indict
    try:
        json_file = open('./class_indices.json', 'r')  # 读取之前保存好的json文件
        class_indict = json.load(json_file)  # 对应的类别信息
    except Exception as e:
        print(e)
        exit(-1)
    predict_class = np.argmax(results)  # 获取概率最大的值所对应的索引
    print(class_indict[str(predict_class)], results[predict_class])  # 得到分类所属类别

if __name__ == "__main__":
    image_path = "/media/omnisky/yanyi_hdd/img_data/tf_dada/"  # 代码所用图集的文件夹
    # train(image_path, batch_size = 6, epochs=50)

    # #测试
    # img_dir = image_path + "val/person/2.jpg"
    # test(img_dir)

    # # 转化
    # # python keras_to_tensorflow-master/keras_to_tensorflow.py --input_model=save_weights/myAlex.h5 --output_model=save_weights/myAlex.pb
    # pd_file_path = 'save_weights/myAlex.pb'
    # print_io_arrays(pd_file_path)
    # # tflite_convert --output_file=save_weights/myAlex.tflite --graph_def_file=save_weights/myAlex.pb --input_arrays=input_1 --output_arrays=softmax/Softmax

    #lite测试
    lite_file_path = 'save_weights/myAlex.tflite'
    img_dir = image_path + "val/person/11.jpg"
    lite_test(lite_model_file=lite_file_path, img_path=img_dir)


