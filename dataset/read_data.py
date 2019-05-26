import os
import numpy as np


def load(path,normalization=False):
  file = open(path,encoding='utf-8')
  x,y=[],[]
  for line in file.readlines():
    label, features = line.strip().split('\t')
    x.append([float(i) for i in features.split(',')])
    y.append([int(float(label))])
  features = np.array(x)
  labels = np.array(y)
  feature_label = np.hstack((features,labels))
  np.random.shuffle(feature_label)

  train, test = feature_label[:int(feature_label.shape[0]*0.7),:], feature_label[int(feature_label.shape[0]*0.7):,:]
  train_x,train_y = train[:,:-1], train[:,-1].astype(int)
  test_x, test_y = test[:,:-1], test[:,-1]
  feat_max, feat_min=0,0
  if normalization:
    train_x, feat_max, feat_min = normalize(train[:,:-1])
    test_x = normalize_testSet(test_x,feat_max,feat_min)

  return (train_x,train_y),(feat_max, feat_min),(test_x, test_y)

def normalize_testSet(test_x,feat_max,feat_min):
  """
  用训练集的最大最小值对测试集进行归一化
  :param test_x:
  :param feat_max:
  :param feat_min:
  :return:
  """
  for col in range(len(feat_max)):
    max = feat_max[col]
    if max != -1:
      # elem = (elem - min) / (max - min)
      test_x[:, col] = (test_x[:, col] - feat_min[col]) / (feat_max[col] - feat_min[col])
  return test_x


def normalize(features):
  '''
  按特征的最大值是否大于1，判断是否要归一化；
  另外，并不是所有的特征位都进行了归一化，因此可以根据返回结果中的 feat_max 中的元素是否为-1，来判断这个特征位是否进行了归一化，是-1，说明没有归一化。
  :param features:
  :return:
  '''
  # 在最小值的基础上扩展1%，以免最小值为0，最大值无所谓
  feat_min = features.min(axis=0) * 0.99
  feat_max = features.max(axis=0)
  for col in range(len(feat_max)):
    max = feat_max[col]
    if max > 1:
      # elem = (elem - min) / (max - min)
      features[:,col] = (features[:,col]-feat_min[col])/(feat_max[col]- feat_min[col])
    else:
      feat_max[col] = -1
  return features,feat_max, feat_min

if __name__ == '__main__':
  features, labels, max, min = load('plus_recommend/plus_recommend_76996', True)
  print(features[1, :])
  a = np.array([[1, 2, 3], [4, 5, 6]])
  a[:, 1] = a[:, 1] - a.min(axis=0)[1]
  print(a)
