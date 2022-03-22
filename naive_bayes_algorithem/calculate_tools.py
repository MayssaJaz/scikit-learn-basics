import numpy as np
def calculate_error(target,predictedDataset):
    p_y=predictedDataset-target
    myArray = np.array(p_y)
    print(f"Percentage of error ---> {np.count_nonzero(myArray)/len(p_y)}")