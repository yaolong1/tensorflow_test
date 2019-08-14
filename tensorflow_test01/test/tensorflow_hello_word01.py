import tensorflow as tf

# ------------------创建图---------------------
# 创建一个常量v1，它是一个1行2列的矩阵
v1 = tf.constant([[2, 3]])

# 创建一个常量v1，它是一个2行1列的矩阵
v2 = tf.constant([[2], [3]])

product = tf.matmul(v1, v2)

print(product)

# -----------------会话-----------------------
sess = tf.Session()

result = sess.run(product)

print(result)

sess.close()

