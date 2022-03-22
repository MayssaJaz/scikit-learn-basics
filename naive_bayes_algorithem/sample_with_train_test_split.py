from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
def sample_with_sklearn(dataset,testSize):
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    D_train, D_test, C_train, C_test= train_test_split(dataset.data, dataset.target, test_size=0.1)
    print (D_train)
    return (D_train, D_test, C_train, C_test)