import numpy as np

class TropicalEconomicDemand:
    """
    Первая в мире открытая реализация тропической модели спроса (Baldwin & Klemperer).
    Моделирует поведение покупателей в пространстве цен через Max-Plus многочлены.
    """
    def __init__(self, bundles, utilities):
        """
        bundles: список доступных наборов товаров (например: [[2, 1], [1, 3]] - 2 яблока и 1 груша)
        utilities: базовая ценность (полезность) каждого набора для покупателя
        """
        self.bundles = np.array(bundles, dtype=float)
        self.utilities = np.array(utilities, dtype=float)

    def compute_agent_utility(self, prices):
        """
        Вычисляет чистую выгоду покупателя для каждого набора при текущих ценах.
        Формула выгоды: Уникальная Полезность - (Количество * Цена)
        В тропическом мире это линейный сдвиг.
        """
        prices = np.array(prices, dtype=float)
        net_utilities = []
        
        for k in range(len(self.utilities)):
            # Затраты на покупку набора: скалярное произведение количества на цены
            cost = np.dot(self.bundles[k], prices)
            # Чистая выгода
            net_utilities.append(self.utilities[k] - cost)
            
        return np.array(net_utilities)

    def predict_market_demand(self, prices):
        """
        Тропический оператор максимума (oplus).
        Определяет индекс набора товаров, который обеспечивает МАКСИМАЛЬНУЮ выгоду.
        Точки, где выбор меняется — это тропические hypersurfaces (границы спроса).
        """
        net_utilities = self.compute_agent_utility(prices)
        # Находим самый выгодный набор для покупателя
        best_bundle_idx = np.argmax(net_utilities)
        return best_bundle_idx, self.bundles[best_bundle_idx]
