from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

class ModelTrainer:
    def __init__(self):
        self.linear_model = LinearRegression()
        self.logistic_model = LogisticRegression(max_iter=1000)

    def train_linear(self, X, y):
        """Train linear regression for goals prediction."""
        self.linear_model.fit(X, y)
        predictions = self.linear_model.predict(X)
        return mean_squared_error(y, predictions)

    def train_logistic(self, X, y):
        """Train logistic regression for match result classification."""
        self.logistic_model.fit(X, y)
        predictions = self.logistic_model.predict(X)
        return accuracy_score(y, predictions)
