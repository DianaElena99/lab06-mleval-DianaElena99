import numpy as np
from collections import Counter

#code for multiclass classification


def BinaryLossFx(real, comp, labels):
    rez = 0
    tot = len(real)
    for r,c in zip(real, comp):
        idx = indexOf(max(c), c)
        if labels[idx] != r:
            rez += 1
    return rez/tot


def indexOf(elem, lista):
    idx = -1
    i = 0
    for e in lista:
        if e == elem:
            idx = i
        i += 1
    return idx

class Classification:
    def __init__(self, realLabels, compLabels, labels):
        self.realLabels = realLabels
        self.compLabels = compLabels
        self.labels = labels
        n = len(self.labels)//2
        self.positives = self.labels[n:]
        self.negatives = self.labels[:n]


    def SoftMax(self):
        asoc = []
        for L in self.labels:
            nr = int("".join(str(x) for x in L), 2)
            asoc.append(nr)

        e_x = np.exp(asoc - np.max(asoc))

        rez = e_x / e_x.sum(axis=0)
        return rez


    """
        Loss function for Multi Class
        real, comp values in arr = softmax()
        
        Loss(real, comp) = sum (( abs(real-comp) * ln(real/comp))
    """
    def LossFx(self):
        loss = 0
        data = self.SoftMax()

        for r,c in zip(self.realLabels, self.compLabels):
            if (r!=c):
                loss += abs(data[indexOf(r, self.labels)] - data[indexOf(c, self.labels)])*np.log(data[indexOf(r, self.labels)]/ data[indexOf(c, self.labels)])
        return loss

    """
    :returns acc - accuracy 
    :returns pp - precision for positive
    :returns rp - recall for positive
    :returns pn - precision for negative
    :returns rn - recall for negative
    """
    def evaluate(self):
        acc = sum([1 if self.realLabels[i]==self.compLabels[i] else 0 for i in range(len(self.realLabels))])/len(self.compLabels)
        acc = round(acc, 2)

        #tp - true positive
        tp = sum([1 if self.realLabels[i] in self.positives and self.compLabels[i] in self.positives else 0 for i in range(len(self.realLabels))])

        #tn - true negative
        tn = sum([1 if self.realLabels[i] in self.negatives and self.compLabels[i] in self.negatives else 0 for i in range(len(self.realLabels))])

        #fp - false positive
        fp = sum([1 if self.realLabels[i] in self.negatives and self.compLabels[i] in self.positives else 0 for i in range(len(self.realLabels))])

        #fn - false negative
        fn = sum([1 if self.realLabels[i] in self.positives and self.compLabels[i] in self.negatives else 0 for i in range(len(self.realLabels))])

        pp = round(tp / (tp + fp), 2)
        rp = round( tp / (tp + fn), 2)

        pn = round( tn / (tn + fn) , 2)
        rn = round( tn / (tn + fp), 2)

        return acc, pp, rp, pn, rn

if __name__ == '__main__':

    #test data for binary classification Loss function
    labels = {0:'spam', 1:'ham'}
    realLabels = ['spam', 'spam', 'ham', 'ham', 'spam', 'ham']

    #outputs comp : spam, ham, ham, spam, spam, ham => 2 din 6 gresite = 0.3333
    computedOutputs = [[0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.9, 0.1], [0.7, 0.3], [0.4, 0.6]]
    print("\nBinary Classif. Loss Fx", BinaryLossFx(realLabels, computedOutputs, labels))


    print("\n MultiClass Classif.")
    rL = [[1,0,0],[0,1,0],[1,0,0],[1,0,0],[0,1,0],[0,0,1]]
    cL = [[0,0,1],[1,0,0],[0,1,0],[1,0,0],[0,1,0],[0,0,1]]

    clas2 = Classification(rL,cL,[[1,0,0],[0,0,1],[0,1,0]])
    rez = clas2.evaluate()
    print("accuracy : ", rez[0], "\npositive prec : ", rez[1], "; positive recall: ", rez[2], "\nnegative prec : ", rez[3], " negative recall : ", rez[4])
    print("\nMultiClass Loss Fx", clas2.LossFx())



"""
    realLabels = ['infected', 'recovered', 'infected', 'infected', 'recovered', 'infected', 'recovered', 'normal', 'normal', 'recovered','normal', 'normal', 'normal',
                  'normal', 'normal', 'recovered', 'normal', 'normal', 'normal', 'normal']
    computedLabels = ['infected','recovered', 'infected', 'normal', 'recovered', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal',
                      'normal', 'normal','recovered','recovered', 'normal', 'normal','recovered', 'normal', 'infected']


    clas = Classification(realLabels, computedLabels, ['infected', 'recovered', 'normal'])
    rez = clas.evaluate()
    print("acc : ", rez[0], "\npos prec : ", rez[1], "; pos recall: ", rez[2], "\nneg prec : ", rez[3], " neg recall : ", rez[4])


"""