import numpy as np
from .basic_ops import TropicalCore

class TropicalOptimizer(TropicalCore):
    """Модуль тропической оптимизации расписаний на основе работ Н. Кривулина."""
    def __init__(self, mode='max'):
        super().__init__(mode)

    def solve_scheduling(self, A, h):
        """Расчет максимально допустимого вектора времени стартов x."""
        n = len(h)
        x = np.zeros(n)
        for i in range(n):
            possible_starts = []
            for j in range(n):
                if A[i, j] != self.zero and i != j:
                    possible_starts.append(h[j] - A[i, j])
                else:
                    possible_starts.append(h[i])
            x[i] = np.min(possible_starts) if self.mode == 'max' else np.max(possible_starts)
        return x

    def validate_system(self, A, x, g=None):
        """Валидация графа ограничений на наличие временных парадоксов."""
        n = len(x)
        if g is None:
            g = np.full(n, float('-inf') if self.mode == 'max' else float('inf'))
        is_consistent = True
        node_errors, edge_errors = set(), set()
        
        for i in range(n):
            if self.mode == 'max' and x[i] < g[i]:
                is_consistent = False
                node_errors.add(i)
                
        for i in range(n):
            for j in range(n):
                if i != j and A[i, j] != self.zero:
                    if self.mode == 'max' and (A[i, j] + x[j] > x[i] + 1e-5):
                        is_consistent = False
                        edge_errors.add((i, j))
                        node_errors.add(i)
                        node_errors.add(j)
        return is_consistent, node_errors, edge_errors
