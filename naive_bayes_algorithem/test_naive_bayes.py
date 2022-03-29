from calculate_tools import calculate_error
from sample_naive_bayes import split
from sample_with_train_test_split import sample_with_sklearn

def test(choice,classifier, dataset):
    if (choice == 1):
        [dataS1, dataS2, targetS1, targetS2] = split(dataset)
    if (choice == 2):
        testSize = 0.22
        [dataS1, dataS2, targetS1, targetS2] = sample_with_sklearn(dataset, testSize)    
    classifier.fit(dataS1, targetS1)  # training on the first sample
    p = classifier.predict(dataS2)  # testing on the second sample
    error=calculate_error(targetS2, p)
    return (error)


def test_k_times(classifier, numberOfTimes, dataset):
    predicted= 0
    choice = 0
    while(choice != 1 and choice != 2):
        choice = int(input("choice 1-sample without using sklearn 2-sample using sklearn"))
    for k in range(numberOfTimes):
        predicted += test(choice,classifier, dataset)
    return (predicted /numberOfTimes)
