import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ============================
# 1. Параметры варианта
# ============================

a = 0.4
b = 0.2
c = 0.07
d = 0.2

INPUT_SIZE = 8
HIDDEN = 3
EPOCHS = 3000
LR = 0.01


# ============================
# 2. Генерация данных
# ============================

def f(x):
    return a * np.cos(b * x) + c * np.sin(d * x)

X_all = np.linspace(0, 10, 300)
y_all = f(X_all)

X, y = [], []

for i in range(len(X_all) - INPUT_SIZE):
    X.append(X_all[i:i + INPUT_SIZE])
    y.append(y_all[i + INPUT_SIZE])

X = np.array(X)
y = np.array(y).reshape(-1, 1)

split = int(0.7 * len(X))
X_train = torch.tensor(X[:split], dtype=torch.float32)
y_train = torch.tensor(y[:split], dtype=torch.float32)
X_test  = torch.tensor(X[split:], dtype=torch.float32)
y_test  = torch.tensor(y[split:], dtype=torch.float32)


# ============================
# 3. Архитектура ИНС (как в методичке)
#     Сигмоида → линейный выход
# ============================

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN)
        self.act = nn.Sigmoid()
        self.fc2 = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.fc2(x)


model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ============================
# 4. Обучение (без ранней остановки)
# ============================

loss_train_hist = []
loss_test_hist = []

for epoch in range(EPOCHS):

    model.train()
    optimizer.zero_grad()

    pred_train = model(X_train)
    loss_train = criterion(pred_train, y_train)
    loss_train.backward()
    optimizer.step()

    loss_train_hist.append(loss_train.item())

    model.eval()
    with torch.no_grad():
        pred_test = model(X_test)
        loss_test = criterion(pred_test, y_test).item()

    loss_test_hist.append(loss_test)

print("Обучение завершено.")


# ============================
# 5. Предсказания
# ============================

model.eval()
with torch.no_grad():
    train_pred = model(X_train).numpy()
    test_pred = model(X_test).numpy()


# ============================
# 6. График 1 — обучение
# ============================

plt.figure(figsize=(8,5))
plt.plot(X_train[:, -1], y_train.numpy(), label="Эталон")
plt.plot(X_train[:, -1], train_pred, "--", label="Прогноз ИНС")
plt.grid(); plt.legend()
plt.title("Прогнозируемая функция на участке обучения")
plt.xlabel("x"); plt.ylabel("y")
plt.show()


# ============================
# 7. График 2 — ошибки
# ============================

plt.figure(figsize=(8,5))
plt.plot(loss_train_hist, label="Ошибка обучения")
plt.plot(loss_test_hist, label="Ошибка тестирования")
plt.yscale("log")
plt.grid(); plt.legend()
plt.title("Изменение ошибки в процессе обучения")
plt.xlabel("итерации"); plt.ylabel("MSE")
plt.show()


# ============================
# 8. График 3 — тест
# ============================

plt.figure(figsize=(8,5))
plt.plot(X_test[:, -1], y_test.numpy(), label="Эталон")
plt.plot(X_test[:, -1], test_pred, "--", label="Прогноз ИНС")
plt.grid(); plt.legend()
plt.title("Результаты прогнозирования на тестовой выборке")
plt.xlabel("x"); plt.ylabel("y")
plt.show()


# ============================
# 9. График 4 — сравнение точек
# ============================

plt.figure(figsize=(6,6))
plt.scatter(y_test.numpy(), test_pred, s=12)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "k--")
plt.grid()
plt.title("Сравнение эталонных и прогнозных значений")
plt.xlabel("Эталонные"); plt.ylabel("Прогноз ИНС")
plt.show()


# ============================
# 10. Текстовый вывод
# ============================

print("="*60)
print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ (первые 10 строк)")
print("="*60)

train_output = np.hstack([
    y_train[:10].numpy(),
    train_pred[:10],
    y_train[:10].numpy() - train_pred[:10]
])

print(f"{'Эталонное':>15} {'Полученное':>15} {'Отклонение':>15}")
for r in train_output:
    print(f"{r[0]:>15.6f} {r[1]:>15.6f} {r[2]:>15.6f}")

print("="*60)
print("РЕЗУЛЬТАТЫ ПРОГНОЗА (первые 10 строк)")
print("="*60)

test_output = np.hstack([
    y_test[:10].numpy(),
    test_pred[:10],
    y_test[:10].numpy() - test_pred[:10]
])

print(f"{'Эталонное':>15} {'Полученное':>15} {'Отклонение':>15}")
for r in test_output:
    print(f"{r[0]:>15.6f} {r[1]:>15.6f} {r[2]:>15.6f}")
