from secrets import choice
from sklearn.datasets import load_iris
from test_naive_bayes import test_k_times
from sklearn.tree import DecisionTreeClassifier
from sklearn_validation import cross_validation_with_sklearn

iris_dataset = load_iris()
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
error=test_k_times(classifier, 2, iris_dataset)
print("The error is ",error)
# applying cross validation with sklearn
cross_validation_with_sklearn(iris_dataset, classifier, 4)
