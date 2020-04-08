"""
class that deals with multiple regression

 yi = w1*x1i + w2*x2i + e
 yi in Y = [y1, y2, ..., yn]^T : result
 x1i, x2i in X = [[x11, x12] ... [x1n, x2n]] : coef
 e - intercept

"""

class MyLinearRegression:
    def __init__(self):
        self.intercept = 0.0
        self.coef = []

    def fit(self, computedOutput):
        pass

    """
    :param x - list like [[x11, x12], ... [xn1, xn2]]
    :returns y - list like [y1, ..., yn]
    """
    def predict(self, x):
        if(isinstance(x, list)):
            return [self.intercept + self.coef[0] * val[0] + self.coef[1]*val[1] for val in x]
        return None


if __name__ == '__main__':
    reg = MyLinearRegression()
    print(reg.predict([[1,2,3],[2,4,6],[13,14,15]]))