import numpy as np
import tensorflow as tf
import os
import cv2
import preprocess as pp
label_dict = {' 0.1':0,' 0.2':1,' 0.5':2,' 1':3,' 2':4,' 5':5,' 10':6,' 50':7,' 100':8}

meta_path = r'./ckpt/CNN.ckpt-4000.meta'

# 加载模型
meta_graph = tf.train.import_meta_graph(meta_path)

# 设置默认图
graph = tf.get_default_graph()

# 获取相关张量
x = graph.get_tensor_by_name("x:0")
drop_rate = graph.get_tensor_by_name("drop_rate:0")
logits = graph.get_tensor_by_name("VGG_11/logits/BiasAdd:0")

# 获取集合RAWS,SHADOWS
RAWS = tf.get_collection('RAWS')
SHADOWS = tf.get_collection('SHADOWS')

# 提取影子
assop = [tf.assign(var1, var2) for var1, var2 in zip(RAWS, SHADOWS)]


with tf.Session() as sess:
    # init
    init = [tf.initialize_all_variables()]
    sess.run(init)

    # 获取影子参数
    saver = tf.train.Saver(SHADOWS)
    saver.restore(sess,tf.train.latest_checkpoint(r'./ckpt'))

    # 使用ema
    sess.run(assop)

    test_dir = r'../rmb_data/test_data'
    test_list = os.listdir(test_dir)
    csv_path = r'../rmb_data/train_face_value_label.csv'

    scores = []
    for test in test_list:
        # 读取图片
        rmb = cv2.imread(os.path.join(test_dir,test))
        rmb = cv2.resize(rmb,(256,128))
        rmb = (rmb/255).reshape([1,128,256,3]).astype(np.float32)
        # 获取结果
        logit = sess.run(logits,feed_dict={x:rmb,drop_rate:0.0})
        index = int(np.argmax(logit,1))

        # 比较
        y_ = pp.Reading_Label_CSV(csv_path,test)
        y_ = label_dict[y_]
        score = np.equal(index,y_)
        scores.append(score)
        print('图片%s识别结果：%s'%(test,score))

    scores = np.array(scores)
    acc = np.sum(scores)/scores.size
    print('全部图片识别率为%.2f'%acc)





