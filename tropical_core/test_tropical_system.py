import pytest
import numpy as np

# Импортируем классы из ваших файлов (укажите правильные названия файлов, если они другие)
# Предположим, первый файл называется tropical_core.py, а второй - economics.py
try:
    from tropical_core import TropicalCore
    from economics import TropicalEconomicDemand
except ImportError:
    # Если запускаете в одной папке, Python найдет их локально
    pass

# =====================================================================
# ТЕСТЫ ДЛЯ КЛАССА TROPICALCORE (Проверяем скрытые баги с NumPy и NaN)
# =====================================================================

def test_tropical_core_scalar_ops():
    """Проверка базовых операций с одиночными числами"""
    core = TropicalCore(mode='max')
    assert core.add(5, 10) == 10
    assert core.mul(5, 10) == 15
    assert core.mul(5, core.zero) == core.zero

def test_tropical_core_numpy_bug():
    """КРИТИЧЕСКИЙ ТЕСТ: Проверяем, падает ли метод mul при подаче массивов NumPy"""
    core = TropicalCore(mode='max')
    a = np.array([1.0, 2.0, float('-inf')])
    b = np.array([3.0, float('-inf'), 5.0])
    
    try:
        result = core.mul(a, b)
        # Если ваш текущий код (с if a == self.zero) дойдет сюда, он упадет с ValueError.
        # Если вы его исправили через np.where, тест успешно пройдет.
        assert isinstance(result, np.ndarray), "Метод mul должен уметь работать с массивами!"
    except ValueError as e:
        pytest.fail(f"Батенька, ИИ вас обманул! Метод mul упал на массивах NumPy: {e}")

def test_build_tropical_matrix_nan_on_diagonal():
    """Тест на ловушку главной диагонали: должен заменять NaN на диагонали на минус бесконечность"""
    core = TropicalCore(mode='max')
    # Матрица, где на диагонали стоит NaN
    bad_matrix = [
        [np.nan, 2.0],
        [3.0, 0.0]
    ]
    
    res = core.build_tropical_matrix(bad_matrix)
    # Элемент [0, 0] обязан стать минус бесконечностью, а не остаться NaN!
    assert res[0, 0] == core.zero, "Ошибка! NaN на главной диагонали проигнорирован!"


# =====================================================================
# ТЕСТЫ ДЛЯ TROPICALECONOMICDEMAND (Проверяем экономическую логику)
# =====================================================================

def test_economic_demand_logic():
    """Проверяем, правильно ли считается чистая выгода покупателя"""
    # Наборы: 1-й набор (2 яблока, 1 груша), 2-й набор (1 яблоко, 3 груши)
    bundles = [[2, 1], [1, 3]]
    # Базовая полезность для каждого набора
    utilities = [100.0, 120.0]
    
    demand = TropicalEconomicDemand(bundles, utilities)
    
    # Задаем текущие цены: яблоко стоит 10, груша стоит 20
    prices = [10.0, 20.0]
    
    # Считаем вручную для теста:
    # Затраты 1: 2*10 + 1*20 = 40. Выгода 1: 100 - 40 = 60
    # Затраты 2: 1*10 + 3*20 = 70. Выгода 2: 120 - 70 = 50
    
    net_utils = demand.compute_agent_utility(prices)
    
    assert np.allclose(net_utils, [60.0, 50.0]), "Формула выгоды (Полезность - Затраты) посчитана неверно!"

def test_predict_market_demand():
    """Проверяем, выбирается ли действительно МАКСИМАЛЬНЫЙ по выгоде набор"""
    bundles = [[2, 1], [1, 3]]
    utilities = [100.0, 120.0]
    demand = TropicalEconomicDemand(bundles, utilities)
    
    prices = [10.0, 20.0] # Выгода первого набора (60) больше, чем второго (50)
    
    best_idx, best_bundle = demand.predict_market_demand(prices)
    
    assert best_idx == 0, "Оператор максимума выбрал не тот наборт товаров!"
    assert np.array_equal(best_bundle, [2, 1]), "Возвращен неверный набор товаров!"
