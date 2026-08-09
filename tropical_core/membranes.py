import numpy as np

class BruhatTitsMembrane:
    """
    Оцифровка раздела 'Bruhat-Tits Buildings & Membranes' статьи Йосвига (2024).
    Рассчитывает решетку точек (мембрану) внутри тропического линейного пространства.
    """
    def __init__(self):
        pass

    def generate_membrane_lattice(self, base_vertex, steps=5, spacing=1.0):
        """
        Генерирует дискретные точки решетки (R-lattices) вокруг базовой тропической вершины.
        Эти точки образуют жесткую 'мембрану' в пространстве Брюа-Титса.
        """
        base_vertex = np.array(base_vertex, dtype=float)
        dim = len(base_vertex)
        lattice_points = []

        # Генерируем сдвиги в тропическом проективном пространстве R^d / R_1
        # Сдвиг всех координат на одно число не меняет точку, важна разность!
        for i in range(-steps, steps + 1):
            for j in range(-steps, steps + 1):
                if dim == 3:
                    shift = np.array([0.0, i * spacing, j * spacing])
                    point = base_vertex + shift
                    lattice_points.append(point)
                    
        return np.array(lattice_points)
