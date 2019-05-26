import tensorflow as tf
import dataset.read_data as read_data

feature_num = 604
vector_size = 5
x_p = tf.placeholder(dtype=tf.float16, shape=[None,feature_num])
y_p = tf.placeholder(dtype=tf.float16,shape=[None,1])
b0 = tf.get_variable(name='constant',dtype=tf.float16,initializer=tf.truncated_normal_initializer(-1,1),shape=[1])
w1 = tf.get_variable(name='weight_4_order1',shape=[feature_num],dtype=tf.float16,initializer=tf.truncated_normal_initializer(-1,1))
V = tf.get_variable(shape=[feature_num,vector_size],name='hidden_vec',dtype=tf.float16,initializer=tf.truncated_normal_initializer(-1,1))
order_1 = tf.add(b0,tf.reduce_sum(tf.multiply(x_p, w1),axis=1))
# 二阶组合公式的实现
order_2 = 0.5*tf.reduce_sum(tf.pow(tf.matmul(x_p, V), 2) - tf.matmul(tf.pow(x_p, 2), tf.pow(V, 2)), axis=1)
prediction = tf.sigmoid(order_1 + order_2)
loss = y_p - prediction
op = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = op.minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  (train_x, train_y), (feat_max, feat_min), (test_x, test_y) = read_data.load('../dataset/plus_recommend/plus_recommend_76996',True)
  i=0
  for example in zip(train_x, train_y):
    _order_1,_order_2, _y,_loss,_ = sess.run(fetches=[order_1,order_2,prediction,loss,train], feed_dict={x_p:[example[0]], y_p:[example[1]]})
    i+=1
    if i%10000 == 0:
      print(_order_1,_order_2,_y,example[1],_loss)

