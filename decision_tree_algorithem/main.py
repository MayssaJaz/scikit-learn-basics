from secrets import choice
from sklearn.datasets import load_iris
from sample_decision_tree import split
from sample_with_train_test_split import sample_with_sklearn
from test_decision_tree import test_k_times
from sklearn.tree import DecisionTreeClassifier
from sklearn_validation import cross_validation_with_sklearn

iris_dataset = load_iris()
nb = DecisionTreeClassifier(criterion='entropy', random_state=0)

choice = 0
while(choice != 1 and choice != 2):
    choice = int(
        input("choice 1-sample without using sklearn 2-sample using sklearn"))

if (choice == 1):
    [dataS1, dataS2, targetS1, targetS2] = split(iris_dataset)
if (choice == 2):
    testSize = 0.22
    [dataS1, dataS2, targetS1, targetS2] = sample_with_sklearn(
        iris_dataset, testSize)

test_k_times(nb, 2, dataS1, dataS2, targetS1, targetS2)
# applying cross validation with sklearn
cross_validation_with_sklearn(iris_dataset, nb, 4)
