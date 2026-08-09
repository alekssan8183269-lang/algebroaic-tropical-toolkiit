import numpy as np
from .basic_ops import TropicalCore

class ClassicalMaxPlusEngine(TropicalCore):
    """
    Оцифровка фундаментальных алгоритмов до-2000-х годов.
    Включает спектральные методы Канингем-Грина (1979) и Бачелли (1992).
    """
    def __init__(self):
        super().__init__(mode='max')

    def tropical_trace(self, matrix):
        """Вычисляет тропический след (tr): oplus-сумма (максимум) диагональных элементов."""
        return np.max(np.diag(matrix))

    def matrix_power(self, matrix, power):
        """Возведение матрицы в тропическую степень (посредством тропического умножения)."""
        if power == 1:
            return matrix
        result = matrix
        for _ in range(power - 1):
            # Используем векторизованное умножение из нашего ядра
            result = self.matrix_mul(result, matrix)
        return result

    def compute_cuninghame_green_eigenvalue(self, standard_matrix):
        """
        Алгоритм Канингем-Грина (1979 г.) для поиска главного собственного значения матрицы.
        Определяет максимальную скорость / пропускную способность циклической системы.
        """
        A = self.build_tropical_matrix(standard_matrix)
        n = A.shape
        traces = []
        
        # Проходим по всем возможным длинам путей от 1 до n
        for k in range(1, n + 1):
            A_k = self.matrix_power(A, k)
            tr_k = self.tropical_trace(A_k)
            
            # В тропической алгебре деление — это обычное вычитание/деление в классическом смысле
            if tr_k != self.zero:
                traces.append(tr_k / k)
                
        # Тропическое сложение (oplus) выбирает максимум из полученных следов
        if not traces:
            return self.zero
        return np.max(traces)
