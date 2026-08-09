import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tropical_core.machine_learning import TropicalRegressor
from tropical_core.fast_ops import numba_tropical_matrix_mul

# --- ТЕСТ 1: Скорость Numba ---
print("Тестируем JIT-ускорение матриц...")
A = np.random.rand(300, 300)
B = np.random.rand(300, 300)
# Первый запуск скомпилирует функцию, второй — отработает мгновенно
res = numba_tropical_matrix_mul(A, B, float('-inf'))
print("Матрицы 300x300 успешно перемножены через Numba!")

# --- ТЕСТ 2: Обучение Tropical Регрессии ---
print("\nОбучаем тропическую модель машинного обучения...")
np.random.seed(42)
X_train = np.random.uniform(1, 10, (100, 2))
# Реальная зависимость — жесткий излом (угловатая функция)
y_train = np.maximum(X_train[:, 0] + 2, X_train[:, 1] - 1) + np.random.normal(0, 0.2, 100)

model = TropicalRegressor()
model.fit(X_train, y_train)

print("Веса тропического полинома успешно вычислены:", model.weights)

# Визуализация предсказаний
X_test = np.random.uniform(1, 10, (50, 2))
y_pred = model.predict(X_test)

plt.figure(figsize=(8, 5))
plt.scatter(y_test_clean := np.maximum(X_test[:, 0] + 2, X_test[:, 1] - 1), y_pred, color='darkmagenta', alpha=0.7)
plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], 'r--', label='Идеальный прогноз')
plt.title("Результат Тропического Машинного Обучения (Предсказание vs Реальность)")
plt.xlabel("Реальные значения")
plt.ylabel("Предсказания модели")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()
