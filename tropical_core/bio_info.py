import numpy as np

class TropicalBioinformatics:
    """Модуль тропической биоинформатики для геометрического анализа ДНК."""
    def __init__(self):
        pass

    def compute_tropical_distance(self, tree_vector_A, tree_vector_B):
        """Тропическая проективная метрика расстояния между геновыми профилями."""
        diff = tree_vector_A - tree_vector_B
        return np.max(diff) - np.min(diff)

    def build_evolutionary_matrix(self, dna_sequences):
        """Строит матрицу попарных мутаций (расстояний Хэмминга) между цепочками ДНК."""
        n = len(dna_sequences)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = sum(1 for a, b in zip(dna_sequences[i], dna_sequences[j]) if a != b)
        return matrix

    def dna_to_trajectory(self, sequence):
        """Превращает буквы ДНК в 2D шаги-координаты."""
        moves = {'A': np.array([1, 0]), 'T': np.array([-1, 0]), 'C': np.array([0, 1]), 'G': np.array([0, -1])}
        current_pos = np.array([0, 0])
        trajectory = [current_pos.copy()]
        for nucleotide in sequence.upper():
            if nucleotide in moves:
                current_pos += moves[nucleotide]
                trajectory.append(current_pos.copy())
        return np.array(trajectory)
