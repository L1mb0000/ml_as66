import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

a, b, c, d = 0.1, 0.1, 0.05, 0.1
n_inputs = 2  
n_hidden = 10  
n_samples = 1000

def target_function(X):
    x1, x2 = X[:, 0], X[:, 1]
    return a * np.cos(2 * np.pi * b * x1) + c * np.sin(2 * np.pi * d * x2)

np.random.seed(42)
periods_x1 = 3
periods_x2 = 2

X = np.zeros((n_samples, n_inputs))
X[:, 0] = np.random.uniform(0, periods_x1 / b, n_samples)
X[:, 1] = np.random.uniform(0, periods_x2 / d, n_samples)
y = target_function(X)

split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.hidden = nn.Linear(n_inputs, n_hidden)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(n_hidden, 1)  
    
    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        return self.output(x)

model = SimpleMLP()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
n_epochs = 1000

losses = []
print("Начало обучения...")

for epoch in range(n_epochs):
    model.train()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    losses.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Эпоха {epoch}, Loss: {loss.item():.6f}")

plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel("Эпоха")
plt.ylabel("Ошибка (MSE)")
plt.title("График изменения ошибки")
plt.grid(True)
plt.show()

model.eval()
with torch.no_grad():
    y_train_pred = model(X_train_tensor).numpy().flatten()
    y_test_pred = model(X_test_tensor).numpy().flatten()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(y_train[:100], 'b-', label='Эталон', linewidth=2)
plt.plot(y_train_pred[:100], 'r--', label='Прогноз', linewidth=1.5)
plt.title('Обучающая выборка')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(y_test, 'b-', label='Эталон', linewidth=2)
plt.plot(y_test_pred, 'r--', label='Прогноз', linewidth=1.5)
plt.title('Тестовая выборка')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
x1_test = np.linspace(0, periods_x1 / b, 500)
x2_fixed = np.mean(X[:, 1])
X_periodic = np.column_stack([x1_test, np.full_like(x1_test, x2_fixed)])
X_periodic_tensor = torch.tensor(X_periodic, dtype=torch.float32)

with torch.no_grad():
    y_periodic_pred = model(X_periodic_tensor).numpy().flatten()

y_periodic_true = target_function(X_periodic)

plt.plot(x1_test, y_periodic_true, 'b-', label='Истинная', linewidth=2)
plt.plot(x1_test, y_periodic_pred, 'r--', label='Прогноз', linewidth=1.5)
plt.title('Проверка периодичности')
plt.xlabel('x1')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mse_periodic = mean_squared_error(y_periodic_true, y_periodic_pred)

print("\nМЕТРИКИ КАЧЕСТВА:")
print(f"MSE на обучении: {mse_train:.6f}")
print(f"MSE на тесте: {mse_test:.6f}")
print(f"MSE на полном периоде: {mse_periodic:.6f}")
print(f"Финальная ошибка: {losses[-1]:.6f}")

results_df = pd.DataFrame({
    'x1': X_test[:, 0],
    'x2': X_test[:, 1],
    'Эталон': y_test,
    'Прогноз': y_test_pred,
    'Ошибка': y_test_pred - y_test
})