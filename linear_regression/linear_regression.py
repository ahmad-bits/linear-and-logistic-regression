import numpy as np


class LinearRegression:

    def __init__(self):
        self.w = None
        self.b = None
        self.cost_history = None
        self.lambda_ = None

    def compute_prediction(self, X):

        return np.dot(X, self.w) + self.b



    def compute_cost(self, X, y):

        m = X.shape[0]

        predictions = self.compute_prediction(X)
        errors = predictions - y

        cost = (1 / (2 * m)) * np.sum(errors ** 2)

        regularization = (self.lambda_ / (2 * m)) * np.sum(self.w ** 2)

        cost = cost + regularization

        return cost

    def compute_gradient(self, X, y):
        m = X.shape[0]

        predictions = self.compute_prediction(X)
        errors = predictions - y

        dj_dw = (1 / m) * np.dot(X.T, errors)
        dj_dw = dj_dw + (self.lambda_ / m) * self.w

        dj_db = (1 / m) * np.sum(errors)

        return dj_dw, dj_db



    def fit(self, X, y, alpha=0.01, num_iters=1000, lambda_=0.1):
        n = X.shape[1]

        self.w = np.zeros(n)
        self.b = 0.0
        self.lambda_ = lambda_

        self.cost_history = []

        for i in range(num_iters):
            dj_dw, dj_db = self.compute_gradient(X, y)

            self.w = self.w - alpha * dj_dw
            self.b = self.b - alpha * dj_db

            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)


    def predict(self, X):
        return self.compute_prediction(X)