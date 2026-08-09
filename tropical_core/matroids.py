import numpy as np

class TropicalMatroidEngine:
    """
    Оцифровка раздела 'Matroids & Bergman Fans' из статьи М. Йосвига (2024).
    Реализует проверку независимости векторов и построение бинарного тропического каркаса.
    """
    def __init__(self):
        self.infty = float('inf')

    def is_linearly_independent(self, matrix_subset):
        """
        Проверяет подмножество векторов на линейную независимость (база матроида).
        В тропическом постоянном случае (0 и inf) это определяет топологическую связность.
        """
        mat = np.array(matrix_subset, dtype=float)
        if mat.shape[0] > mat.shape[1]:
            return False
        # Проверка ранга матрицы для классического lift-матроида
        return np.linalg.matrix_rank(mat) == mat.shape[0]

    def build_bergman_fan_vectors(self, standard_matrix):
        """
        Генерирует опорные тропические векторы веера Бергмана.
        Переводит обычные веса в бинарные тропические координаты (0 или inf).
        """
        matrix = np.array(standard_matrix, dtype=float)
        bergman_matrix = np.zeros_like(matrix)
        
        # Матроидный фильтр шума: все значимые связи переходят в 0, отсутствие - в inf
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] == 0 or np.isnan(matrix[i, j]):
                    bergman_matrix[i, j] = self.infty
                else:
                    bergman_matrix[i, j] = 0.0 # Постоянный коэффициент по Йосвигу
                    
        return bergman_matrix
