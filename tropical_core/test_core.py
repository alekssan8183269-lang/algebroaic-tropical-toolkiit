import numpy as np
import pytest
from tropical_core import TropicalCore


def test_numpy_array_safety():
    """Проверяем, что метод mul больше не падает при подаче массивов NumPy"""
    core = TropicalCore(mode="max")

    # Имитируем сложную матричную строку с бесконечностями
    a = np.array([1.0, 2.0, float("-inf")])
    b = np.array([3.0, float("-inf"), 5.0])

    # Старый ИИ-код тут бы упал. Наш новый код отработает как швейцарские часы!
    res = core.mul(a, b)

    # Ожидаем: [1+3, -inf, -inf] -> [4.0, -inf, -inf]
    assert res[0] == 4.0
    assert res[1] == float("-inf")
    assert res[2] == float("-inf")
    print("\n[СИСАДМИНСКИЙ КОНТРОЛЬ ПРОЙДЕН]: Массивы NumPy обработаны без багов!")


def test_nan_on_diagonal_fix():
    """Проверяем, что ловушка с NaN на главной диагонали полностью ликвидирована"""
    core = TropicalCore(mode="max")

    bad_matrix = [[np.nan, 5.0], [0.0, np.nan]]

    res = core.build_tropical_matrix(bad_matrix)

    # Проверяем, что NaN на диагонали превратился в минус бесконечность, а не остался NaN
    assert res[0, 0] == float("-inf")
    assert res[1, 1] == float("-inf")
    # Проверяем, что обычный ноль вне диагонали тоже ушел в бесконечность
    assert res[1, 0] == float("-inf")
    print("[СИСАДМИНСКИЙ КОНТРОЛЬ ПРОЙДЕН]: Все ловушки диагонали и NaN зачищены!")
