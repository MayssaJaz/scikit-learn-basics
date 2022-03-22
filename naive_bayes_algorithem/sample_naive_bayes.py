import random
from sklearn import datasets
import numpy as np
def split(dataset):
  data=dataset.data
  target=dataset.target
  n = len(data)
  indexes = np.random.choice(range(len(data)),len(data))
  #We split our dataset into two parts: 2/3 => training 1/3 =>training
  dataS1 = [data[index] for index in indexes[:int(2/3 * n)]]
  dataS2 = [data[index] for index in indexes[int(1/3 * n):-1]]
  targetS1 = [target[index] for index in indexes[:int(2/3 * n)]]
  targetS2 = [target[index] for index in indexes[int(1/3 * n):-1]]
  return (dataS1,dataS2,targetS1,targetS2)