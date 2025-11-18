import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd


b = 0.5
c = 0.05
d = 0.5

n_inputs = 8
n_hidden = 3


def generate_series(a, N=2000):
    i = np.arange(N)
    y = a * np.cos(b * i) + c * np.sin(d * i)
    return y


def create_dataset(series, look_back=8):
    X, Y = [], []
    for i in range(look_back, len(series)):
        X.append(series[i - look_back:i])
        Y.append(series[i])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# 3. MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        return self.output(x)


# 4. Подбор a: от 0.1 до 0.5 с шагом 0.05
a_values = np.arange(0.1, 0.51, 0.05)
best_a = None
min_test_mse = float('inf')
results = []

print("🔍 Поиск оптимального a...")
print("-" * 60)

for a in a_values:
    y_full = generate_series(a, N=2000)

    X, Y = create_dataset(y_full, look_back=n_inputs)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    X_train_t = torch.tensor(X_train)
    Y_train_t = torch.tensor(Y_train).unsqueeze(1)
    X_test_t = torch.tensor(X_test)
    Y_test_t = torch.tensor(Y_test).unsqueeze(1)

    model = MLP(n_inputs, n_hidden)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for epoch in range(1500):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, Y_train_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        pred_test = model(X_test_t).numpy().flatten()

    test_mse = float(np.mean((Y_test - pred_test) ** 2))
    results.append({'a': round(a, 2), 'test_mse': test_mse})

    if test_mse < min_test_mse:
        min_test_mse = test_mse
        best_a = round(a, 2)
        best_model = model
        best_losses = losses
        best_split_data = (X_train, Y_train, X_test, Y_test, pred_test)

    print(f"a = {a:.2f} → Test MSE = {test_mse:.8f}")

print("-" * 60)
print(f"✅ Оптимальное a = {best_a} (Test MSE = {min_test_mse:.8f})")


# 5. Результаты для best_a
a = best_a
X_train, Y_train, X_test, Y_test, pred_test = best_split_data

# Обучение на лучших данных (ещё раз, чтобы получить предсказания на обучении)
y_full = generate_series(a, N=2000)
X, Y = create_dataset(y_full)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

X_train_t = torch.tensor(X_train)
Y_train_t = torch.tensor(Y_train).unsqueeze(1)
X_test_t = torch.tensor(X_test)
Y_test_t = torch.tensor(Y_test).unsqueeze(1)

model = best_model
with torch.no_grad():
    pred_train = model(X_train_t).numpy().flatten()

# 6.1 График функции (первые 200 точек)
plt.figure(figsize=(10, 3))
y_plot = generate_series(a, N=200)
plt.plot(y_plot, label=f'y[i] = {a}·cos({b}·i) + {c}·sin({d}·i)', color='steelblue')
plt.title('Участок временного ряда для обучения (первые 200 точек)')
plt.xlabel('i')
plt.ylabel('y[i]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 6.2 График ошибки обучения
plt.figure(figsize=(8, 4))
plt.plot(best_losses, color='darkorange')
plt.title(f'Изменение ошибки (MSE) при обучении (a = {best_a})')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.yscale('log')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6.3 Таблица: обучение (первые 10)
train_df = pd.DataFrame({
    'Эталонное значение': Y_train[:10],
    'Полученное значение': pred_train[:10],
    'Отклонение': Y_train[:10] - pred_train[:10]
})
print("\n=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ (первые 10) ===")
print(train_df.round(6).to_string(index=False))

# 6.4 Таблица: прогнозирование (первые 10)
test_df = pd.DataFrame({
    'Эталонное значение': Y_test[:10],
    'Полученное значение': pred_test[:10],
    'Отклонение': Y_test[:10] - pred_test[:10]
})
print("\n=== РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ (первые 10) ===")
print(test_df.round(6).to_string(index=False))

# 6.5 Метрики
train_mse = np.mean((Y_train - pred_train) ** 2)
test_mse = np.mean((Y_test - pred_test) ** 2)
train_mae = np.mean(np.abs(Y_train - pred_train))
test_mae = np.mean(np.abs(Y_test - pred_test))

print(f"\n📊 Оценка при a = {best_a}:")
print(f"Train → MSE: {train_mse:.8f}, MAE: {train_mae:.8f}")
print(f"Test  → MSE: {test_mse:.8f}, MAE: {test_mae:.8f}")


plt.figure(figsize=(10, 4))
plt.plot(Y_test[:100], label='Эталон (тест)', color='blue')
plt.plot(pred_test[:100], label='Прогноз (тест)', color='red', linestyle='--')
plt.title('Сравнение эталона и прогноза (первые 100 точек теста)')
plt.xlabel('Номер точки в тесте')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()