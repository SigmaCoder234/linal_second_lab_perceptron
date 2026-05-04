import numpy as np

class Perceptron:
    def __init__(self, alpha=0.0, gamma=0.9):
        # Веса (w) и смещение (b) инициализируем в методе fit
        self.w = None
        self.b = None
        self.alpha = alpha # Бонус - параметр для L2 регуляризации
        self.gamma = gamma # Бонус - параметр для ускорения сходимости (Momentum)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):

        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

    def compute_loss(self, y_true, y_pred):

        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        log_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) + self.alpha*np.sum(self.w**2) # Бонус - L2 регуляризация
        return log_loss

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)    

    def fit(self, X, y, X_val = None, y_val = None,
            epochs=100, lr=0.1, batch_size=32):

        d = X.shape[1]
        self.w = np.random.randn(d) * 0.01
        self.b = 0.0
        v_w = np.zeros(d)
        v_b = 0.0
        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            permutations = np.random.permutation(X.shape[0])
            X_shuffled = X[permutations]
            y_shuffled = y[permutations]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                y_pred = self.forward(X_batch)
                m = X_batch.shape[0]
                error = y_pred - y_batch

                dw = (1/m) * np.dot(X_batch.T, error) + self.alpha * self.w * 2
                db = (1/m) * np.sum(error)

                # Бонус - моментум
                v_w = self.gamma * v_w + lr * dw
                v_b = self.gamma * v_b + lr * db

                self.w -= v_w
                self.b -= v_b

            train_losses.append(self.compute_loss(y, self.forward(X)))
            
            if X_val is not None and y_val is not None:
                epoch_val_loss = self.compute_loss(y_val, self.forward(X_val))
                test_losses.append(epoch_val_loss)

        # Возвращаем списки, когда все эпохи закончились
        return train_losses, test_losses