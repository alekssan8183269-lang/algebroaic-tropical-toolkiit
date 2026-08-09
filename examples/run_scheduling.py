import sys
import os
import numpy as np
# Позволяет импортировать модули из папки выше
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tropical_core.optimization import TropicalOptimizer
from tropical_core.visualizer import TropicalVisualizer

# Инициализируем данные
opt = TropicalOptimizer(mode='max')
standard_matrix = [[0, 4, 0], [0, 0, 3], [0, 0, 0]]
h_vector = [10.0, 14.0, 20.0]

# Математика и Валидация
A = opt.build_tropical_matrix(standard_matrix)
x = opt.solve_scheduling(A, h_vector)
ok, n_err, e_err = opt.validate_system(A, x)

print("Тропический таймлайн расписания:", x)

# Запускаем методы визуализации для логистики (Методы 1, 2, 3, 4, 5, 7, 10)
print("Отрисовка Метод 1...")
TropicalVisualizer.method_1_network_graph(A, x, h_vector, n_err, e_err, {0,1}, {(0,1)}, "Логистическая Сеть")
print("Отрисовка Метод 2...")
TropicalVisualizer.method_2_balance_fan(delay=4.0, deadline=14.0)
print("Отрисовка Метод 3...")
TropicalVisualizer.method_3_crystal_surface(delay=4.0, deadline=14.0)
print("Отрисовка Метод 4...")
TropicalVisualizer.method_4_gradient_field(delay=4.0)
print("Отрисовка Метод 5...")
TropicalVisualizer.method_5_contour_waves()
print("Отрисовка Метод 7...")
TropicalVisualizer.method_7_crystal_growth([1, -1, 0], [0, 1, -1], [2, -1, 3])
print("Отрисовка Метод 10...")
TropicalVisualizer.method_10_pareto_shield([20, 40, 60, 80], [45, 30, 20, 10], [20, 40, 60, 80], [45, 30, 20, 10])
