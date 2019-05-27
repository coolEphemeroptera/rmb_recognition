import tensorflow as tf
from tensorflow.contrib.layers import flatten
import pickle
import os
import numpy as np
import tfr_tools as tfr
import cv2

# 生成相关目录保存生成信息
def GEN_DIR():
    import os
    if not os.path.isdir('ckpt'):
        print('文件夹ckpt未创建，现在在当前目录下创建..')
        os.mkdir('ckpt')
    if not os.path.isdir('trainLog'):
        print('文件夹trainLog未创建，现在在当前目录下创建..')
        os.mkdir('trainLog')

# 保存变量至集合
def saving_into_collection(name,vars):
    for var in vars:
        tf.add_to_collection(name,var)
    
# 保存训练记录
def Saving_Train_Log(filename,var,dir=r'./trainLog'):
    var = np.array(var)
    f = open(os.path.join(dir,filename),'wb')
    pickle.dump(var,f)
    f.close()
    print('成功保存记录：%s!'%filename)


# 显示
def show(img):
    cv2.namedWindow(' ', cv2.WINDOW_NORMAL)
    cv2.imshow(' ', img)
    cv2.waitKey(0)

# 定义VGG_16
def VGG_16(x,drop_rate,reuse=False):

    with tf.variable_scope('VGG_16', reuse=reuse):
        # block1
        with tf.name_scope('block_1'):
            conv1_1 = tf.layers.conv2d(x,64,3,padding='same',activation=tf.nn.relu,
                                       kernel_initializer=tf.variance_scaling_initializer,name='conv1_1')
            conv1_2 = tf.layers.conv2d(conv1_1, 64, 3, padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.variance_scaling_initializer, name='conv1_2')
            pool1 = tf.layers.max_pooling2d(conv1_2,2,2)

        # block2
        with tf.name_scope('block_2'):
            conv2_1 = tf.layers.conv2d(pool1, 128, 3, padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.variance_scaling_initializer, name='conv2_1')
            conv2_2 = tf.layers.conv2d(conv2_1, 128, 3, padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.variance_scaling_initializer, name='conv2_2')
            pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2)

        # block3
        with tf.name_scope('block_3'):
            conv3_1 = tf.layers.conv2d(pool2, 256, 3, padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.variance_scaling_initializer, name='conv3_1')
            conv3_2 = tf.layers.conv2d(conv3_1, 256, 3, padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.variance_scaling_initializer, name='conv3_2')
            conv3_3 = tf.layers.conv2d(conv3_2, 256, 3, padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.variance_scaling_initializer, name='conv3_3')
            pool3 = tf.layers.max_pooling2d(conv3_3, 2, 2)

        # block4
        with tf.name_scope('block_4'):
            conv4_1 = tf.layers.conv2d(pool3, 512, 3, padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.variance_scaling_initializer, name='conv4_1')
            conv4_2 = tf.layers.conv2d(conv4_1, 512, 3, padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.variance_scaling_initializer, name='conv4_2')
            conv4_3 = tf.layers.conv2d(conv4_2, 512, 3, padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.variance_scaling_initializer, name='conv4_3')
            pool4 = tf.layers.max_pooling2d(conv4_3, 2, 2)

        # block5
        with tf.name_scope('block_5'):
            conv5_1 = tf.layers.conv2d(pool4, 512, 3, padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.variance_scaling_initializer, name='conv5_1')
            conv5_2 = tf.layers.conv2d(conv5_1, 512, 3, padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.variance_scaling_initializer, name='conv5_2')
            conv5_3 = tf.layers.conv2d(conv5_2, 512, 3, padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.variance_scaling_initializer, name='conv5_3')
            pool5 = tf.layers.max_pooling2d(conv5_3, 2, 2)

        # FC6
        flat = flatten(pool5)
        FC6 = tf.layers.dense(flat,2048,activation=tf.nn.relu,kernel_initializer=tf.variance_scaling_initializer,name='FC6')
        DP6 = tf.nn.dropout(FC6,rate=drop_rate,name='DP6')

        # FC7
        FC7 = tf.layers.dense(DP6, 1024, activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer,name='FC7')
        DP7 = tf.nn.dropout(FC7, rate=drop_rate,name='DP7')

        # FC8
        logits = tf.layers.dense(DP7, 9,  kernel_initializer=tf.variance_scaling_initializer,name='logits')

        return logits

# ---------------------------------------------- 计算图 ------------------------------------------------------------- #
GEN_DIR()
# 定义输入
x = tf.placeholder(tf.float32, [None, 128, 256, 3], 'x')
y_ = tf.placeholder(name="y_", shape=[None, 9], dtype=tf.float32)
drop_rate = tf.placeholder(dtype=tf.float32,name='drop_rate')
STEP = tf.Variable(0, trainable=False)

# 定义损失函数
# 计算交叉熵
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=VGG_16(x,drop_rate),name='cross_entropy')

# 梯度下降
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy,global_step=STEP)  # 使用adam优化器来以0.0001的学习率来进行微调

# 准确率测试
correct_prediction = tf.equal(tf.argmax(VGG_16(x,drop_rate,True), 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')

# --------------------------------------------- 滑动平均 ---------------------------------------------------------- #
# 训练参数
vars = [var for var in tf.trainable_variables()]
saving_into_collection('RAWS',vars)
# 使用EMA
EMA = tf.train.ExponentialMovingAverage(decay=0.99,num_updates=STEP)
ema_op = EMA.apply(vars) # apply ema
# 保存影子
shadows = [EMA.average(var) for var in vars]
saving_into_collection('SHADOWS',shadows)
# 确认更新
with tf.control_dependencies([train_step,ema_op]):
    train_opt_ema = tf.no_op(name='train_opt_ema')
# 保存模型
saver = tf.train.Saver(var_list = vars+shadows)

# --------------------------------------------- 迭代 ---------------------------------------------------------#
epochs = 20
batch_size = 64
data_size = int(39620*0.7)
max_iters = int(data_size*epochs/batch_size)

# 读取单个tfr
[idx,data,label] = tfr.Reading_TFR(r'./train_tfr/rmb-*',isShuffle=False,datatype=tf.uint8,labeltype=tf.float64)

# 读取成批tfr
[idx_batch,data_batch,label_batch] = tfr.Reading_Batch_TFR(idx,data,label,isShuffle=False,batchSize=batch_size,data_size=128*256*3,
                                                       label_size=9)

with tf.Session() as sess:
    # 初始化变量
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    # 开启协调器
    coord = tf.train.Coordinator()
    # 启动线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 迭代
    ACC = []
    LOST = []
    for steps in range(max_iters):

        # 读取数据集
        [idxs, datas, labels] = sess.run([idx_batch, data_batch, label_batch])
        # 格式修正
        datas = (datas.reshape([-1, 128, 256, 3]) / 255).astype(np.float32)
        labels = labels.reshape([-1, 9]).astype(np.float32)
        # 显示
        # for img,lb in zip(datas,labels):
        #     print(lb)
        #     imshow(img)

        # 测试和训练
        if steps%5==0:
            acc = sess.run(accuracy,feed_dict={x:datas,y_:labels,drop_rate:0.0})
            ACC.append([steps,acc])
            print('iters:%d/%d..acc:%.2f'%(steps,max_iters,acc))
        else:
            sess.run(train_opt_ema,feed_dict={x:datas,y_:labels,drop_rate:0.2})

        # 保存损失函数
        lost = sess.run(cross_entropy,feed_dict={x:datas,y_:labels,drop_rate:0.2})
        LOST.append([steps,lost])

        # 保存模型
        if steps % 500 == 0 and steps > 0:
            saver.save(sess, './ckpt/CNN.ckpt', global_step=steps)

    # 保存
    Saving_Train_Log('ACC', ACC)
    Saving_Train_Log('LOST', LOST)


    # 关闭线程
    coord.request_stop()
    coord.join(threads)
