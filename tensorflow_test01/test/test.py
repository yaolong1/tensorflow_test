import tensorflow as tf
import numpy as np # A
# -------------------test1-------------
# m1 = [[1.0, 2.0],
#     [3.0, 4.0]] # B
#
# m2 = np.array([[1.0, 2.0],
#     [3.0, 4.0]], dtype=np.float32) # B
#
# m3 = tf.constant([[1.0, 2.0],
#     [3.0, 4.0]]) # B
#
# print(type(m1)) # C
# print(type(m2)) # C
# print(type(m3)) # C
#
# t1 = tf.convert_to_tensor(m1, dtype=tf.float32) # D
# t2 = tf.convert_to_tensor(m2, dtype=tf.float32) # D
# t3 = tf.convert_to_tensor(m3, dtype=tf.float32) # D
#
# print(type(t1)) # E
# print(type(t2)) # E
# print(type(t3)) # E

# -------------------test2-------------
# A 定义一个任意张量
# B 对张量取负
# C 打印对象
# x = tf.constant([[1, 2]]) #A
# neg_x = tf.negative(x) #B
# print(neg_x) #C

# 显式地指定会话
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# print(sess.run(neg_x))

# 隐式地指定会话
# sess = tf.InteractiveSession()
# print(neg_x.eval())
# -----------------------test2-------------
# raw_data = np.random.normal(10, 1, 10000) #A
#
# alpha = tf.constant(0.05) #B
# curr_value = tf.placeholder(tf.float32) #C
# prev_avg = tf.Variable(0.) #D
# update_avg = alpha * curr_value + (1 - alpha) * prev_avg
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(len(raw_data)): #E
#         curr_avg = sess.run(update_avg, feed_dict={curr_value: raw_data[i]})
#         sess.run(tf.assign(prev_avg, curr_avg))
#         print(raw_data[i], curr_avg)

# -----------------------test3 tensorBoard测试-------------
raw_data = np.random.normal(10, 1, 100)

alpha = tf.constant(0.05)
curr_value = tf.placeholder(tf.float32)
prev_avg = tf.Variable(0.)
update_avg = alpha * curr_value + (1 - alpha) * prev_avg

avg_hist = tf.summary.scalar("running_average", update_avg) #A
value_hist = tf.summary.scalar("incoming_values", curr_value) #B
merged = tf.summary.merge_all() #C
writer = tf.summary.FileWriter("./logs") #D
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # sess.add_graph(sess.graph) #E
    for i in range(len(raw_data)):
        summary_str, curr_avg = sess.run([merged, update_avg], feed_dict={curr_value:
        raw_data[i]}) #F
        sess.run(tf.assign(prev_avg, curr_avg))
        print(raw_data[i], curr_avg)
        writer.add_summary(summary_str, i)