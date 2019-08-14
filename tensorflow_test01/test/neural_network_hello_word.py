import tensorflow as tf
import numpy as np

#创建数据


x_data = np.random.rand(100).astype(np.float32)
# 0.3和0.4是预测的数据
y_data = x_data*0.3 + 0.4

# 创建tensorflow的结构（图）----start------------------#

# 权重
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# 偏执
biases = tf.Variable(tf.zeros([1]))

y = x_data*Weights + biases

# 预测y_data与y的差别（reduce_mean表示求误差） 计算损失值
loss = tf.reduce_mean(tf.square(y-y_data))
# 创建GradientDescentOptimizer梯度下降优化器 学习效率为0.5<1
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 减少loss值让误差值变小，来到达训练目的
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 创建tensorflow的结构（图）----end----------------------#

# 新建会话
with tf.Session() as sess:
    sess.run(init)
    # 开始训练 训练201次
    for step in range(201):
        sess.run(train)
        if step % 10 == 0:
            print(step, sess.run(Weights), sess.run(biases))










