from calculate_tools import calculate_error


def test(classifier, dataS1, dataS2, targetS1, targetS2):
    classifier.fit(dataS1, targetS1)  # training on the first sample
    p = classifier.predict(dataS2)  # testing on the second sample
    calculate_error(targetS2, p)
    return (p)


def test_k_times(classifier, numberOfTimes, dataS1, dataS2, targetS1, targetS2):
    for k in range(numberOfTimes):
        predicted = test(classifier, dataS1, dataS2, targetS1, targetS2)
