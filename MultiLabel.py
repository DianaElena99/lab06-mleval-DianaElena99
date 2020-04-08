import numpy as np
from collections import Counter
from sklearn import linear_model

def ListDiff(a,b):
    diff = 0
    for xa, xb in zip(a,b):
        if xa!=xb:
            diff+=1
    return diff

from sklearn.metrics import hamming_loss
class MultiLabel:
    #inputs : a list like [param1, param2, param3 ... , param n]
    #compOutputs, realOutputs : list like [[0,1,,1...0],[],[],...,[]]
    #                           where len(innerList) = NoOfLabels
    #                           0 -> doesn't have that label
    #                           1 -> has label

    def __init__(self, inputs, compOutputs, realOutputs):
        self.inputs = inputs
        self.compOutputs = compOutputs
        self.realOutputs = realOutputs

    # a) using sklearn
    def HammingLoss(self):
        return hamming_loss(np.array(self.realOutputs), np.array(self.compOutputs))

    # b) native code
    def MyLossFx(self):
        rez = 0
        tot = 2*len(self.realOutputs[0]) * len(self.realOutputs)
        for r,c in zip(self.realOutputs, self.compOutputs):
            if r!=c:
                rez += ListDiff(self.realOutputs, self.compOutputs)
        return rez/tot

if __name__ == '__main__':

    print("MultiLabel Classification\n 1.Loss function implemented in sklearn\n 2.Native code")
    multi = MultiLabel([2,4,5,8],[[1,0,1],[1,0,0],[0,1,0],[1,0,1]],[[1,0,1],[1,0,0],[0,1,1],[1,0,0]])
    print("\n1. ",multi.HammingLoss())
    print("2. ", multi.MyLossFx())

    multi2 = MultiLabel([13,14,25,81,10,100], [[1,1,0,0],[1,0,0,0],[0,1,1,0],[1,0,0,1],[1,0,1,1]], [[0,1,0,0],[1,0,0,0],[0,0,1,0],[1,0,0,1],[1,0,1,1]])
    print("\n1. ", multi2.HammingLoss())
    print("2. ", multi2.MyLossFx())