from sklearn.model_selection import cross_val_score


def cross_validation_with_sklearn(dataset, classifieur, numberOfFolds):
    scores = [cross_val_score(classifieur, dataset.data, dataset.target, cv=iteration)
              for iteration in range(3, numberOfFolds)]
    errors = [1 - score for score in scores]
    print("errors -->", errors)
