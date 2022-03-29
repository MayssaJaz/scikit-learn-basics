from secrets import choice
from sklearn import datasets
from sklearn import naive_bayes
from collections import Counter
import pylab as pl
from itertools import cycle
import numpy as np


def show_data_target(dataset):
    print("=====================data=======================")
    '''.data: stores an array of dimensions n*m where n is the number of instances, and m is the number
    of attributes.'''
    print(dataset.data)
    print("=====================target=======================")
    '''.target:stores the classes (labels) of each instance (in the supervised case).'''
    print(dataset.target)


def plot_2D(data, target, target_names):
    colors = cycle('rgbcmykw')  # cycle of colors
    target_ids = range(len(target_names))
    pl.figure()  # Create a new figure, or activate an existing figure.
    for i, c, label in zip(target_ids, colors, target_names):
        pl.scatter(data[target == i, 2],
                   data[target == i, 3], c=c, label=label)
  # coordinates of two points to draw a to seperate between the first class and the other two classes
    point1 = [0, 2]
    point2 = [3, 0]
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    pl.plot(x_values, y_values)
    pl.legend()
    pl.show()


def predict(nb,dataset):
    # training stage: we are training our model on the whole dataset except the last element
    nb.fit(dataset.data[:-1], dataset.target[:-1])
    # predicting the class of the 32th instance
    p31 = nb.predict([dataset.data[31]])
    print(p31)
    # predicting the class of the last instance
    plast = nb.predict([dataset.data[-1]])
    print(plast)
    # predicting the class of the whole dataset
    p = nb.predict(dataset.data[:])
    print(p)
    return (p)


def calculate_error(dataset, predictedDataset):
    choice = 0
    while(choice != 1 and choice != 2):
        choice = int(input("Choose: 1-first method, 2-second method"))
    if (choice == 1):
        ea = 0
        for i in range(len(dataset.data)):
            if (predictedDataset[i] != dataset.target[i]):
                ea = ea + 1
        print(f"Percentage of error (1st method)--> {ea/len(dataset.data)}")
    if (choice == 2):
        p_y = predictedDataset-dataset.target
        myArray = np.array(p_y)
        print(
            f"Percentage of error (2nd method)--> {np.count_nonzero(myArray)/len(p_y)}")


def calculate_accuracy(nb,dataset):
    accuracy = nb.score(dataset.data, dataset.target)
    print(f"Percentage of accuracy (3rd method)--> {accuracy}")


if  __name__ == '__main__':
    # Importing the iris dataset
    iris_dataset = datasets.load_iris()
    show_data_target(iris_dataset)
    plot_2D(iris_dataset.data, iris_dataset.target, iris_dataset.target_names)
    # we choose the algorithem we are going to wrok with:Naive Bayes
    nb = naive_bayes.MultinomialNB(fit_prior=True)
    predictedDataset = predict(nb,iris_dataset)
    calculate_error(iris_dataset, predictedDataset)
    calculate_accuracy(nb,iris_dataset)