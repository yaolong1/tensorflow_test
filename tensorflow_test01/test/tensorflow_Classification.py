import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
mnist = read_data_sets('MNIST_data', one_hot=True)


# 添加一层神经层 inputs表示输入的节点数据，in_size：表示输入的节点个数, out_size：输出节点个数，activation_function表激活函数
def add_layer(inputs, in_size, out_size,activation_function=None):
    # 定义权重
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
    # 定义偏执
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='B')
    # 将权重和偏执组合成一个公式
    Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases, name='Wx_plus_b')
    if activation_function is None:
        outputs = Wx_plus_b
        return outputs
    else:
        # 利用激活函数
        outputs = activation_function(Wx_plus_b)
        # ***tensorboard显示输出值outputs的直方图
        return outputs
# 计算准确性
def compute_accuracy(v_xs, v_ys):
    global prediction
    # 运算的预测值结果
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    # 将运算出的预测值 和 真实的比较是否相同
    correct_prediction = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(v_ys, 1))
    # 计算数据多少个是对的多少个是错的(精度)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 预测精度结果值
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# 定义两个占位符
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

# 添加输出层
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# 预测与实际数据之间的误差
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))# loss


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 初始化变量

init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 运行初始化
    sess.run(init)

    # 训练1000次
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if i % 20 == 0:
            print(compute_accuracy(mnist.test.images, mnist.test.labels))

