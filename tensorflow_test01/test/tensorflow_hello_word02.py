import tensorflow as tf

# ------------------创建图---------------------
# 创建变量
num = tf.Variable(0, name="count")
# 创建一个加法操作，把当前数字加一
new_Value = tf.add(num, 10)
# 创建一个赋值操作，把new_Value赋值给num
op = tf.assign(num, new_Value)

# -------------------创建会话------------------

with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 打印最初num值
    print(sess.run(num))
    for i in range(5):
        sess.run(op)
        print(sess.run(num))
