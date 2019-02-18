#!/usr/bin/python

import numpy as np
def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    largest = 0.0
    index_large = 0
    
    for itr in range(1,10):
        for index in range(0,len(predictions)):
            error = predictions[index] - net_worths[index]
            if error > largest:
                largest = error
                index_large = index
                
        predictions = np.delete(predictions,index_large,0)
        ages = np.delete(ages,index_large,0)
        net_worths = np.delete(net_worths,index_large,0)
        largest = 0.0
        index_large = 0
        
    for index in range(0,len(predictions)):
        error = predictions[index] - net_worths[index]
        cleaned_data.append((ages[index],net_worths[index],error))
      
            
        
    
    
    return cleaned_data

