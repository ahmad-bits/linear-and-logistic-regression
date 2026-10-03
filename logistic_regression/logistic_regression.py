import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None
        self.cost_history = None


    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))



    def cost(self, X, y):
        m = X.shape[0]

        z = np.dot(X, self.w) + self.b
        predictions = self.sigmoid(z)

        # Clip predictions to prevent log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)

        cost = -(1 / m) * np.sum(
            y * np.log(predictions) +
            (1 - y) * np.log(1 - predictions)
        )

        return cost



    def gradient(self, X, y):
        m = X.shape[0]

        z = np.dot(X, self.w) + self.b
        predictions = self.sigmoid(z)

        error = predictions - y

        dw = (1 / m) * np.dot(X.T, error)
        db = (1 / m) * np.sum(error)

        return dw, db



    def fit(self, X, y):

        m, n = X.shape

        self.w = np.zeros(n)
        self.b = 0.0
        self.cost_history = []

        for i in range(self.iterations):

            dw, db = self.gradient(X, y)

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            self.cost_history.append(self.cost(X, y))



    def predict_probability(self, X):

        z = np.dot(X, self.w) + self.b

        return self.sigmoid(z)



    def predict(self, X):

        probabilities = self.predict_probability(X)

        return (probabilities >= 0.5).astype(int)