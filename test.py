import numpy as np
import tensorflow as tf

def get_accuracy_from_trained_model(meta_path,test_x):

    # 加载模型
    meta_graph = tf.train.import_meta_graph(meta_path)

    # 设置默认图
    graph = tf.get_default_graph()

    # 获取相关张量
    x = graph.get_tensor_by_name("x")
    y_ = graph.get_tensor_by_name("y_")
    accuracy = graph.get_tensor_by_name("accuracy")

    # 获取集合RAWS,SHADOWS
    RAWS = tf.get_collection('RAWS')
    SHADOWS = tf.get_collection('SHADOWS')

    # 提取影子
    assop = [tf.assign(var1, var2) for var1, var2 in zip(RAWS, SHADOWS)]

    with tf.Session() as sess:
        # init
        init = [tf.initialize_all_variables()]
        sess.run(init)

        # 获取参数
        saver = tf.train.Saver(RAWS,SHADOWS)
        saver.restore(sess,tf.train.latest_checkpoint(r'./ckpt'))

        # 使用ema
        sess.run(assop)

        # 返回测试结果
        return sess.run(accuracy,feed_dict={x:test_x})
