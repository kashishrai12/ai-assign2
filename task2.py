import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# =========================
# LOAD DATASET
# =========================
DATA_DIR = r"C:\Users\raika\OneDrive\Desktop\ai lab assign 2\dataset"
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("No CSV file found in dataset directory.")

print("Using dataset file:", csv_files[0])
data = pd.read_csv(os.path.join(DATA_DIR, csv_files[0]))

# Clean column names
data.columns = data.columns.str.upper().str.strip()

# Handle missing values
data = data.fillna(data.mean())

# =========================
# FEATURE SELECTION
# =========================
X = data[['CRIM', 'RM']].values
y = data[['MEDV']].values

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Scale target for stability
y = y / 50.0

# =========================
# TRAIN-TEST SPLIT
# =========================
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))

X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

# =========================
# ACTIVATION FUNCTIONS
# =========================
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# =========================
# LOSS FUNCTION
# =========================
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# =========================
# MLP MODEL
# =========================
class MLP:
    def __init__(self, optimizer='gd', lr=0.001, beta=0.9):
        self.lr = lr
        self.optimizer = optimizer
        self.beta = beta

        # Weight initialization
        self.W1 = np.random.randn(2, 5) * 0.1
        self.b1 = np.zeros((1, 5))
        self.W2 = np.random.randn(5, 3) * 0.1
        self.b2 = np.zeros((1, 3))
        self.W3 = np.random.randn(3, 1) * 0.1
        self.b3 = np.zeros((1, 1))

        self.v = {}
        self.m = {}
        self.t = 0

        for p in ['W1','b1','W2','b2','W3','b3']:
            self.v[p] = np.zeros_like(getattr(self, p))
            self.m[p] = np.zeros_like(getattr(self, p))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)

        self.y_hat = self.a2 @ self.W3 + self.b3
        return self.y_hat

    def backward(self, X, y):
        m = len(y)
        dL = (self.y_hat - y) / m

        dW3 = self.a2.T @ dL
        db3 = np.sum(dL, axis=0, keepdims=True)

        da2 = dL @ self.W3.T
        dz2 = da2 * relu_derivative(self.z2)

        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(self.z1)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2,
            'W3': dW3, 'b3': db3
        }

        # Gradient clipping
        for g in grads:
            grads[g] = np.clip(grads[g], -1, 1)

        self.update(grads)

    def update(self, grads):
        if self.optimizer == 'gd':
            for p in grads:
                setattr(self, p, getattr(self, p) - self.lr * grads[p])

        elif self.optimizer == 'momentum':
            for p in grads:
                self.v[p] = self.beta * self.v[p] + self.lr * grads[p]
                setattr(self, p, getattr(self, p) - self.v[p])

        elif self.optimizer == 'adam':
            self.t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for p in grads:
                self.m[p] = beta1 * self.m[p] + (1 - beta1) * grads[p]
                self.v[p] = beta2 * self.v[p] + (1 - beta2) * (grads[p] ** 2)

                m_hat = self.m[p] / (1 - beta1 ** self.t)
                v_hat = self.v[p] / (1 - beta2 ** self.t)

                setattr(self, p,
                        getattr(self, p) - self.lr * m_hat / (np.sqrt(v_hat) + eps))

    def train(self, X, y, epochs=1000):
        losses = []
        start = time.time()
        for _ in range(epochs):
            y_pred = self.forward(X)
            losses.append(mse(y, y_pred))
            self.backward(X, y)
        return losses, time.time() - start

# =========================
# TRAIN & COMPARE OPTIMIZERS
# =========================
optimizers = ['gd', 'momentum', 'adam']
plt.figure()

for opt in optimizers:
    print(f"\nOptimizer: {opt.upper()}")
    model = MLP(optimizer=opt, lr=0.001)
    losses, t = model.train(X_train, y_train)

    y_test_pred = model.forward(X_test)
    test_mse = mse(y_test, y_test_pred)

    print(f"Final Train Loss: {losses[-1]:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Training Time: {t:.2f}s")

    plt.plot(losses, label=opt)

plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.show()

# =========================
# PREDICTED vs ACTUAL
# =========================
y_test_pred_rescaled = y_test_pred * 50
y_test_rescaled = y_test * 50

plt.scatter(y_test_rescaled, y_test_pred_rescaled)
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Predicted vs Actual House Prices")
plt.plot([0, 50], [0, 50])
plt.show()
