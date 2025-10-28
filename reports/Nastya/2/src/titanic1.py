import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка локального CSV-файла
df = pd.read_csv("california_housing.csv")  # Убедись, что файл лежит рядом с этим скриптом

# Проверка и очистка данных
df.dropna(subset=["median_house_value", "median_income"], inplace=True)

# Выбор признака и целевой переменной
X = df[["median_income"]]
y = df["median_house_value"]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказания
y_pred = model.predict(X_test)

# Метрики
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.5, label="Фактические значения")
plt.plot(X_test, y_pred, color="red", label="Линия регрессии")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Доход vs Стоимость жилья")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
