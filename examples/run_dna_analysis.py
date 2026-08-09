import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tropical_core.bio_info import TropicalBioinformatics
from tropical_core.visualizer import TropicalVisualizer

bio = TropicalBioinformatics()
dna_h = "ATCGATCGATCGGGCCCAAATTT"
dna_c = "ATCGATCGATCAGGCCCAAATTT" # Мутация в 12 позиции

# Генерируем траектории
t_h = bio.dna_to_trajectory(dna_h)
t_c = bio.dna_to_trajectory(dna_c)

# Ищем точки мутаций для подсветки
mut_p = []
for i in range(min(len(t_h), len(t_c))):
    if not np.array_equal(t_h[i], t_c[i]): mut_p.append(t_h[i])

# Запускаем специфические методы визуализации ДНК (Методы 6, 8, 9)
print("Отрисовка Метод 6...")
TropicalVisualizer.method_6_dna_trajectory(t_h, t_c, "Человек", "Шимпанзе", np.array(mut_p))

print("Отрисовка Метод 8...")
grid_data = np.zeros((len(dna_h), len(dna_c)))
for i in range(len(dna_h)):
    for j in range(len(dna_c)):
        grid_data[i, j] = abs(i - j) if dna_h[i] != dna_c[j] else 0
TropicalVisualizer.method_8_dna_drift_grid(grid_data)

print("Отрисовка Метод 9...")
angles = np.linspace(0, 2*np.pi, len(dna_h))
r1 = np.ones(len(dna_h)) * 10
r2 = np.ones(len(dna_h)) * 10
r2[12:] += 3 # Имитация сдвига мутации на радаре
TropicalVisualizer.method_9_polar_radar(angles, r1, r2, "Человек", "Мутант")
