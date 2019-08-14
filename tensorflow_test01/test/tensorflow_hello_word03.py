import tensorflow as tf

# ------------------创建图---------------------
# 创建占位符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# 将两个占位符相乘，不计算
new_Value = tf.multiply(input1, input2)

# -------------------创建会话------------------
with tf.Session() as sess:
    a = sess.run(new_Value, feed_dict={input1: 12.0, input2: 19.0})
    print(a)