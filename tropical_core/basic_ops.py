import numpy as np

class TropicalCore:
    """Математическое ядро для полуколец Max-Plus (mode='max') и Min-Plus (mode='min')."""
    def __init__(self, mode='max'):
        if mode not in ['max', 'min']:
            raise ValueError("Режим должен быть 'max' или 'min'")
        self.mode = mode
        self.zero = float('-inf') if mode == 'max' else float('inf')

    def add(self, a, b):
        """Тропическое сложение (oplus): выбор max или min."""
        return np.maximum(a, b) if self.mode == 'max' else np.minimum(a, b)

    def mul(self, a, b):
        """Тропическое умножение (otimes): обычное сложение."""
        if (a == self.zero) or (b == self.zero):
            return self.zero
        return a + b

    def build_tropical_matrix(self, standard_matrix):
        """Конвертер обычной матрицы в тропическую."""
        matrix = np.array(standard_matrix, dtype=float)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i != j and (matrix[i, j] == 0 or np.isnan(matrix[i, j])):
                    matrix[i, j] = self.zero
        return matrix
