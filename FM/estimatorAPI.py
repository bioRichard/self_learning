import tensorflow as tf
import dataset.read_data as read_data
import os

def del_file(path):
  ls = os.listdir(path)
  for i in ls:
    c_path = os.path.join(path, i)
    if os.path.isdir(c_path):
      del_file(c_path)
    else:
      os.remove(c_path)

def train_input_fn(features, labels, batch_size):
  dataset = tf.data.Dataset.from_tensor_slices(({'features':features},labels))

  # Shuffle, repeat, and batch the examples.
  dataset = dataset.shuffle(1000).repeat().batch(batch_size)

  # Return the dataset.
  return dataset


def eval_input_fn(features, labels, batch_size):
  """for evaluation or prediction"""
  if labels is None:
    # No labels, use only features.
    dataset = tf.data.Dataset.from_tensor_slices({'features': features})
  else:
    dataset = tf.data.Dataset.from_tensor_slices(({'features': features}, labels))

  assert batch_size is not None, "batch_size must not be None"
  dataset = dataset.batch(batch_size)
  return dataset


def my_model_fn(features, labels, mode, params):
  feature_num = 604
  vector_size = 20
  x_p = features['features']
  y_p = labels
  b0 = tf.get_variable(name='constant', dtype=tf.float32, initializer=tf.truncated_normal_initializer(-1, 1), shape=[1])
  w1 = tf.get_variable(name='weight_4_order1', shape=[feature_num], dtype=tf.float32,
                       initializer=tf.truncated_normal_initializer(-1, 1))
  V = tf.get_variable(shape=[feature_num, vector_size], name='hidden_vec', dtype=tf.float32,
                      initializer=tf.truncated_normal_initializer(-1, 1))
  order_1 = tf.add(b0, tf.reduce_sum(tf.multiply(x_p, w1), axis=1))
  # 二阶组合公式的实现
  order_2 = 0.5 * tf.reduce_sum(tf.pow(tf.matmul(x_p, V), 2) - tf.matmul(tf.pow(x_p, 2), tf.pow(V, 2)), axis=1)
  prediction = tf.sigmoid(order_1 + order_2)
  print(prediction.get_shape)
  loss = tf.losses.log_loss(labels=labels, predictions=prediction)

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode, predictions=prediction)
  if mode == tf.estimator.ModeKeys.EVAL:
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=prediction,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

  assert mode == tf.estimator.ModeKeys.TRAIN
  op = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  train_op = op.minimize(loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def base_model_fn(features, labels, mode, params):
  x_p = features['features']
  labels = tf.cast(tf.reshape(labels, shape=[-1,1]), tf.int32)
  print(labels.get_shape)
  l1 = tf.layers.dense(inputs=x_p,units=300,activation=tf.sigmoid)
  l2 = tf.layers.dense(inputs=l1,units=300,activation=tf.sigmoid)

  logits = tf.layers.dense(inputs=l2,units=2,activation=tf.sigmoid)
  print(logits.get_shape)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits =logits)

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode, predictions=logits)
  if mode == tf.estimator.ModeKeys.EVAL:
    predicted_classes = tf.argmax(logits, 1)
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

  assert mode == tf.estimator.ModeKeys.TRAIN
  op = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  train_op = op.minimize(loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def main():
  # 1、训练数据
  (train_x, train_y), (feat_max, feat_min), (test_x, test_y) = read_data.load('../dataset/plus_recommend/plus_recommend_76996', False)
  # 2、运行相关参数
  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  del_file('output/base')
  run_config = tf.estimator.RunConfig(
    model_dir='output/base',
    save_checkpoints_steps=1000,
    session_config=session_config)
  # 3、训练
  classifier = tf.estimator.Estimator(
    model_fn=base_model_fn,
    config=run_config
    )
  classifier.train(input_fn=lambda:train_input_fn(train_x,train_y,64), steps=2*train_x.shape[0]/64)
  # classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, 64), steps=10)
  eval_result = classifier.evaluate( input_fn=lambda:eval_input_fn(test_x, test_y, 64))
  print(eval_result)

if __name__ == '__main__':
  main()