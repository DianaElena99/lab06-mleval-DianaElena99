from math import sqrt

import matplotlib.pyplot as plt


class RegressionProblem:
    """
    ins : collection of input data
    outs : collection of computed outputs
    realOuts : collection of real outputs
    """
    def __init__(self, ins, outs, realOuts):
        self.ins = ins
        self.outs = outs
        self.realOuts = realOuts


    """
    computes Accuracy, Precission, Recall
    """
    def APR(self):
        tp = 0
        linComputed = sum(self.outs, [])
        linReal = sum(self.realOuts, [])
        for x,y in zip(linComputed, linReal):
            if x!=y:
              tp += 1
        return tp/len(linReal)

    """
    :returns e1 : error computed as sum of differences
    :returns e2 : error computed as sum of squares
    
    """

    def computeError(self):
        e1 = sum([sum([abs(x-y) for x,y in zip(r,c)])/len(r) for r,c in zip(self.realOuts, self.outs)])
        e1 /= len(self.ins)
        e2 = sum([sum([(x-y)**2 for x,y in zip(r,c)])/(len(r)**2) for r,c in zip(self.realOuts, self.outs)])
        e2 /= len(self.ins)
        return e1, sqrt(e2)

    def plotOutputs(self):
        indexes = [i for i in range(len(self.outs))]
        real, = plt.plot(indexes, self.realOuts, 'ro', label='Real Outputs')
        computed, = plt.plot(indexes, self.outs, 'bo', label='Computed Outputs')
        plt.xlim(0,int(max(self.ins))+1)
        plt.ylim(0,int(max(self.outs))+1)
        plt.legend([real, (real, computed)], ["Real", "Computed"])
        plt.show()


#test program

if __name__ == '__main__':
    reg = RegressionProblem([[1,1],[2,1],[3,3],[5,7]], [[2,3,6],[4,7,9],[6,10,30],[10,70,80]], [[2,3,5.9],[4.2, 7, 8.1],[6.5, 10, 29],[9.9, 73, 81]])
    err = reg.computeError()
    #reg.plotOutputs()
    print("Error (E1) : ", round(err[0],2), "\n Error (E2) : ", round(err[1],2))
    print("Accuracy: ", reg.APR())

