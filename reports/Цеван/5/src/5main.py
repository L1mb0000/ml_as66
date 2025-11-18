import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

# Fix random seeds
np.random.seed(42)
torch.manual_seed(42)

# Variant 12 parameters
a = 0.4
b = 0.6
c = 0.06
d = 0.6

INPUT_SIZE = 10
HIDDEN_SIZE = 4

def generate_series(N=800):
    """
    Example nonlinear model for time series.
    Replace the formula inside the loop with your exact lab formula
    if you have a different one.
    """
    y = np.zeros(N, dtype=float)
    u = np.random.uniform(-1.0, 1.0, size=N)

    y[0] = 0.0
    y[1] = 0.0

    for k in range(2, N):
        y[k] = (
            a * y[k-1] / (1.0 + y[k-1]**2)
            + b * y[k-2]
            + c * np.sin(u[k-1])
            + d * u[k-2]
        )
    return y

def make_dataset(series, window=INPUT_SIZE):
    X, T = [], []
    for k in range(window, len(series)):
        X.append(series[k-window:k])
        T.append(series[k])
    X = np.array(X, dtype=np.float32)
    T = np.array(T, dtype=np.float32).reshape(-1, 1)
    return X, T

def smooth(y, window=7):
    if window <= 1:
        return y
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, kernel, mode="same")

# ----- data generation -----
series = generate_series(N=800)
X, T = make_dataset(series, window=INPUT_SIZE)

train_size = int(0.7 * len(X))
X_train, T_train = X[:train_size], T[:train_size]
X_test,  T_test  = X[train_size:], T[train_size:]

X_train_t = torch.from_numpy(X_train)
T_train_t = torch.from_numpy(T_train)
X_test_t  = torch.from_numpy(X_test)
T_test_t  = torch.from_numpy(T_test)

# ----- neural network definition -----
class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net(INPUT_SIZE, HIDDEN_SIZE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# ----- training -----
epochs = 500
loss_history = []

for _ in range(epochs):
    net.train()
    optimizer.zero_grad()
    outputs = net(X_train_t)
    loss = criterion(outputs, T_train_t)
    loss.backward()
    optimizer.step()
    loss_history.append(float(loss.item()))

# ----- prediction -----
net.eval()
with torch.no_grad():
    train_pred = net(X_train_t).numpy().flatten()
    test_pred  = net(X_test_t).numpy().flatten()

train_true = T_train.flatten()
test_true  = T_test.flatten()

train_residuals = train_pred - train_true
test_residuals  = test_pred - test_true

# ----- plots -----

# 1. Training: smoothed true vs prediction
plt.figure(figsize=(16, 8))
plt.plot(smooth(train_true, 9), label="True (train, smooth)", linewidth=3)
plt.plot(smooth(train_pred, 9), label="Prediction (train, smooth)", linewidth=3)
plt.title("Training section: true vs prediction (smoothed)", fontsize=18)
plt.xlabel("Sample index", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Test: smoothed true vs prediction
plt.figure(figsize=(16, 8))
plt.plot(smooth(test_true, 9), label="True (test, smooth)", linewidth=3)
plt.plot(smooth(test_pred, 9), label="Prediction (test, smooth)", linewidth=3)
plt.title("Test section: true vs prediction (smoothed)", fontsize=18)
plt.xlabel("Sample index", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Training loss (smoothed)
plt.figure(figsize=(16, 6))
plt.plot(smooth(np.array(loss_history), 15), linewidth=3)
plt.title("Training loss (MSE, smoothed)", fontsize=18)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("MSE", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Residuals train
plt.figure(figsize=(16, 6))
plt.plot(train_residuals, linewidth=1.5)
plt.title("Residuals on training set", fontsize=18)
plt.xlabel("Sample index", fontsize=14)
plt.ylabel("Error", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Residuals test
plt.figure(figsize=(16, 6))
plt.plot(test_residuals, linewidth=1.5)
plt.title("Residuals on test set", fontsize=18)
plt.xlabel("Sample index", fontsize=14)
plt.ylabel("Error", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ----- tables for report -----
train_table = pd.DataFrame({
    "True": train_true,
    "Pred": train_pred,
    "Error": train_residuals
})

test_table = pd.DataFrame({
    "True": test_true,
    "Pred": test_pred,
    "Error": test_residuals
})

print("First 10 rows: TRAIN")
print(train_table.head(10))
print("\nFirst 10 rows: TEST")
print(test_table.head(10))
