# Change the below two lines to run a custom testcase and check
# --------------------------
A = np.array([1, 1, 1, 1, 1, 1]) # Input the array elements by changing the values inside the list
NewDataValue = 1 # Enter the New data value that should be added to the array of numbers given already
# --------------------------

import numpy as np
def UpdateMean(OldMean, NewDataValue, n, A):
    newMean = (OldMean*n + NewDataValue) / (n+1)
    return newMean
def UpdateStd(OldMean, OldStd, NewMean, NewDataValue, n, A):
    newStd = pow((pow(OldStd,2)*(n-1)+pow(NewDataValue,2)+n*pow(OldMean,2)-(n+1)*pow(NewMean,2))/n,0.5)
    return newStd
def UpdateMedian(OldMedian, NewDataValue, n, A):
    temp = np.array(sorted(A)) # If we don't assume A to be sorted 
    if ( n % 2 == 0 ):
        if ( NewDataValue > temp[n//2] ):
            return temp[n//2]
        elif ( NewDataValue > temp[(n//2)-1] ):
            return temp[(n//2)-1]
        else:
            return NewDataValue    
    else:
        if ( NewDataValue > temp[(n+1)//2] ):
            return (OldMedian+temp[(n+1)//2])/2
        elif ( NewDataValue < temp[(n-3)//2] ):
            return (OldMedian+temp[(n-3)//2])/2
        else:
            return (OldMedian+NewDataValue)/2

OldMean = np.mean(A)
OldMedian = np.median(A)
OldStd = np.std(A)
n = len(A)
newMean = UpdateMean(OldMean, NewDataValue, n, A)
newMedian = UpdateMedian(OldMedian, NewDataValue, n, A)
newStd = UpdateStd(OldMean, OldStd, newMean, NewDataValue, n, A)
print("New Mean =",newMean)
print("New Median =",newMedian)
print("New Standard Deviation =",newStd)
