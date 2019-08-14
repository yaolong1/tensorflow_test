import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# ###########将直线你合成二次方程 y_data#############
# 添加一层神经层 inputs表示输入的节点数据，in_size：表示输入的节点个数, out_size：输出节点个数，activation_function表激活函数
def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weignts'):

            # 定义权重
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')

            # ***tensorboard显示权重Weights的直方图
            tf.summary.histogram(layer_name+'/weignts',Weights)
        with tf.name_scope('biases'):

            # 定义偏执
            biases = tf.Variable(tf.zeros([1, out_size])+0.1, name='B')

            # ***tensorboard显示偏执biases的直方图
            tf.summary.histogram(layer_name + '/biases', biases)

        with tf.name_scope('Wx_plus_b'):

            # 将权重和偏执组合成一个公式
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases, name='Wx_plus_b')

        if activation_function is None:
            outputs = Wx_plus_b
            return outputs
        else:
            with tf.name_scope('outputs'):
                # 利用激活函数
                outputs = activation_function(Wx_plus_b)
                # ***tensorboard显示输出值outputs的直方图
                tf.summary.histogram(layer_name + '/outputs', outputs)
                return outputs

# 制造数据 300个[-1，1]之间的数据，并且将它们维度华 1维三百行
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]

# 让y_data变得更像一个真实的数据
noise = np.random.normal(0, 0.05, x_data.shape)

# 组合数据 y_data
y_data = np.square(x_data) - 0.5 + noise

# 定义两个占位符
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_input")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_input")

# ##输入层向隐藏层add_layer()输入数据###
# 输入层传入数据xs，数据输入为1个节点（因为只有传了一个数据xs），输出为10个节点（因为使用tf.nn.relu激活函数隐藏层的节点为了10个）
# 第三个参数是使用激活函数tf.nn.relu
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)


# ##隐藏层向输出层传递数据###
# 隐藏层传入数据l1，数据输入为10个节点（隐藏层有10个节点），输出为1个节点（只有一个结果prediction）
# 因为只有一层隐藏层所以输出不会有激活函数 即激活函数为activation_function=None
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

with tf.name_scope('loss'):

    # 损失值的计算
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))


    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):

    # 训练准备
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()
with tf.Session() as sess:

    # ***将以上的直方图合并
    merged = tf.summary.merge_all()

    # **运行流程图
    writer = tf.summary.FileWriter("logs/", sess.graph)

    # 运行初始化
    sess.run(init)

    # * 结果可视化视图准备
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()

    # 训练1000次
    for i in range(1000):
        # 开始训练
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 20 == 0:
            # 计算loss值
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

            # ***运行它
            result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            # ***总结写入 i是折线图的横坐标，隔20表示一格
            writer.add_summary(result, i)

            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            # *当前的预测值
            prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
            # *将当前的预测值结果绘图
            lines = ax.plot(prediction_value, 'r-', lw=5)
            # *每0.1秒绘制一次
            plt.pause(0.1)
    # 结果可视化显示
    plt.ioff()
    plt.show()
    plt.pause(0)