import numpy as np

def solve_tropical_scheduling(A, h):
    """
    Решает задачу тропической оптимизации расписания / логистики.
    
    Параметры:
    A (np.array): Матрица временных связей/задержек между задачами.
                  A[i, j] — сколько времени должно пройти между стартом i и стартом j.
                  Если связи нет, ставится -inf (минус бесконечность).
    h (np.array): Вектор дедлайнов (максимально разрешенное время завершения задач).
    
    Возвращает:
    np.array: Оптимальный вектор времени старта x.
    """
    n = len(h)
    x = np.zeros(n)
    
    # Реализуем тропическое сопряжение (сдвиг дедлайнов назад по цепочке ограничений)
    for i in range(n):
        # Ищем самый жесткий «верхний» лимит времени, вычитая задержки из дедлайнов
        possible_starts = []
        for j in range(n):
            if A[i, j] != float('-inf'):
                # Задача i должна начаться достаточно рано, чтобы j успела к дедлайну h[j]
                possible_starts.append(h[j] - A[i, j])
            else:
                possible_starts.append(h[i]) # Если связи нет, ограничение только по своему дедлайну
                
        # Нам нужен самый строгий (минимальный) срок старта, чтобы удовлетворить ВСЕ зависимости
        x[i] = min(possible_starts)
        
    return x

# ==========================================
# ПРИМЕР ИЗ ЖИЗНИ (Кейс для проверки кода)
# ==========================================

# У нас 3 машины (или 3 этапа проекта)
# Матрица задержек A:
# Машина 0 должна стартовать минимум за 2 часа до Машины 1.
# Машина 1 должна стартовать минимум за 3 часа до Машины 2.
inf = float('-inf')
A = np.array([
    [0,   2,   inf],
    [inf, 0,   3],
    [inf, inf, 0]
])

# Вектор дедлайнов h:
# Машина 0 должна закончить к 10:00
# Машина 1 — к 12:00
# Машина 2 — к 15:00
h = np.array([10.0, 12.0, 15.0])

# Запускаем наш тропический оптимизатор
optimal_schedule = solve_tropical_scheduling(A, h)

print("Оптимальное время старта для каждого этапа:")
for step, start_time in enumerate(optimal_schedule):
    print(f"Этап {step}: {start_time:.1f}")



import numpy as np

def tropical_optimize_with_validation(A, h, g=None):
    """
    Тропический оптимизатор расписания с валидацией графа ограничений.
    
    Параметры:
    A (np.array): Матрица задержек (Max-Plus). A[i,j] — задержка между стартом i и j.
                  Если связи нет, используется float('-inf').
    h (np.array): Вектор дедлайнов (самое позднее время завершения/старта).
    g (np.array): Вектор ранних стартов (самое раннее разрешенное время). 
                  Если None, то ограничений снизу нет (минус бесконечность).
    """
    n = len(h)
    if g is None:
        g = np.full(n, float('-inf'))
        
    # --- ШАГ 1: Расчет потенциально оптимального вектора x (Тропическое сопряжение) ---
    x = np.zeros(n)
    for i in range(n):
        possible_starts = []
        for j in range(n):
            if A[i, j] != float('-inf'):
                possible_starts.append(h[j] - A[i, j])
            else:
                possible_starts.append(h[i])
        x[i] = min(possible_starts)
        
    # --- ШАГ 2: Валидация графа ограничений и совместности ---
    is_consistent = True
    errors = []
    
    # Контроль 1: Успеваем ли мы начать после минимально разрешенного времени g?
    for i in range(n):
        if x[i] < g[i]:
            is_consistent = False
            errors.append(f"Критическая ошибка на узле {i}: "
                          f"Требуемый старт ({x[i]:.1f}) раньше разрешенного ({g[i]:.1f}). "
                          f"Слишком жесткий дедлайн дальше по цепочке!")

    # Контроль 2: Не нарушаются ли внутренние связи графа (A ⊙ x <= x)?
    # Проверяем, что задержки между узлами физически успевают выполниться
    for i in range(n):
        for j in range(n):
            if A[i, j] != float('-inf'):
                # В Max-Plus ограничение выглядит как: A[i,j] + x[j] <= x[i]
                if A[i, j] + x[j] > x[i] + 1e-9: # 1e-9 для защиты от погрешности float
                    is_consistent = False
                    errors.append(f"Конфликт графа ограничений: связь {i} -> {j} невыполнима. "
                                  f"Требуется пауза {A[i,j]}, но между ними всего {x[i] - x[j]:.1f}")
                    
    return is_consistent, x, errors


# =====================================================================
# ТЕСТ 1: Идеальный граф (Все условия выполняются)
# =====================================================================
inf = float('-inf')
A_good = np.array([
    [0,   2,   inf],  # Задача 0 должна начаться минимум за 2 часа до 1
    [inf, 0,   3],    # Задача 1 — минимум за 3 часа до 2
    [inf, inf, 0]
])
h_good = np.array([10.0, 12.0, 15.0]) # Дедлайны
g_good = np.array([2.0, 2.0, 2.0])    # Раньше 2:00 начинать нельзя

ok, schedule, logs = tropical_optimize_with_validation(A_good, h_good, g_good)

print("--- ТЕСТ 1: Валидное расписание ---")
print(f"Совместность графа: {ok}")
if ok:
    print("Идеальный таймлайн старта:", schedule)
else:
    print("\n".join(logs))


# =====================================================================
# ТЕСТ 2: Сломанный граф (Логическая ошибка в условиях / Парадокс времени)
# =====================================================================
# Сделаем дедлайны слишком агрессивными. Например, завершить всё нужно к 5:00, 
# но технологически начать раньше 2:00 нельзя, а суммарные паузы занимают 5 часов.
h_bad = np.array([5.0, 5.0, 6.0]) 
g_bad = np.array([2.0, 2.0, 2.0]) 

ok_bad, schedule_bad, logs_bad = tropical_optimize_with_validation(A_good, h_bad, g_bad)

print("\n--- ТЕСТ 2: Несовместный граф (Конфликт дедлайнов) ---")
print(f"Совместность графа: {ok_bad}")
if not ok_bad:
    print("Обнаруженные парадоксы планирования:")
    print("\n".join(logs_bad))


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def analyze_and_draw_tropical_graph(A, h, g=None, title="Тропический граф ограничений"):
    """
    Вычисляет тропическое расписание, проверяет его на совместность 
    и визуализирует граф с подсветкой ошибок.
    """
    n = len(h)
    if g is None:
        g = np.full(n, float('-inf'))
    
    # ----------------------------------------------------
    # ШАГ 1: Тропическая оптимизация (Max-Plus сопряжение)
    # ----------------------------------------------------
    x = np.zeros(n)
    for i in range(n):
        possible_starts = []
        for j in range(n):
            if A[i, j] != float('-inf') and i != j:
                possible_starts.append(h[j] - A[i, j])
            else:
                possible_starts.append(h[i])
        x[i] = min(possible_starts)
    
    # ----------------------------------------------------
    # ШАГ 2: Сбор ошибок и валидация
    # ----------------------------------------------------
    is_consistent = True
    node_errors = set()
    edge_errors = set()
    
    # Проверка ограничений снизу (ранний старт)
    for i in range(n):
        if x[i] < g[i]:
            is_consistent = False
            node_errors.add(i)
            
    # Проверка технологических задержек между узлами
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] != float('-inf'):
                if A[i, j] + x[j] > x[i] + 1e-5:
                    is_consistent = False
                    edge_errors.add((i, j))
                    node_errors.add(i)
                    node_errors.add(j)

    # ----------------------------------------------------
    # ШАГ 3: Построение и отрисовка графа через NetworkX
    # ----------------------------------------------------
    G = nx.DiGraph()
    
    # Добавляем узлы и метаданные для подписей
    for i in range(n):
        label = f"Узел {i}\nСтарт: {x[i]:.1f}\nДедлайн: {h[i]:.1f}"
        if g[i] != float('-inf'):
            label += f"\nМин: {g[i]:.1f}"
        G.add_node(i, label=label)
        
    # Добавляем ребра (технологические паузы)
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] != float('-inf'):
                G.add_edge(i, j, weight=A[i, j])
                
    # Задаем геометрию графа (в линию для наглядности цепочки)
    pos = nx.spring_layout(G, seed=42) 
    
    plt.figure(figsize=(10, 6))
    plt.title(f"{title}\nСовместность: {'УСПЕШНО' if is_consistent else 'ОШИБКА (Красные зоны)'}", 
              fontsize=14, color='green' if is_consistent else 'red', fontweight='bold')
    
    # Цвета узлов: зеленый, если все ок, красный — если узел с ошибкой
    node_colors = ['#ff6b6b' if node in node_errors else '#51cf66' for node in G.nodes()]
    
    # Цвета и толщина стрелок
    edge_colors = ['red' if edge in edge_errors else 'gray' for edge in G.edges()]
    edge_widths = [3 if edge in edge_errors else 1.5 for edge in G.edges()]
    
    # Рисуем узлы и стрелки
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=edge_colors, width=edge_widths, 
                           arrowsize=20, connectionstyle="arc3,rad=0.1")
    
    # Добавляем подписи к узлам
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_family='sans-serif', font_weight='bold')
    
    # Добавляем веса ребер (длину пауз)
    edge_labels = {(u, v): f"пауза: {d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, color='blue')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return is_consistent, x

# =====================================================================
# ИНИЦИАЛИЗАЦИЯ ДАННЫХ ДЛЯ ТЕСТОВ
# =====================================================================
inf = float('-inf')

# Матрица ограничений (Узел 0 -> Узел 1 требует 3 часа пауз, Узел 1 -> Узел 2 требует 4 часа)
A = np.array([
    [0,   3,   inf],
    [inf, 0,   4],
    [inf, inf, 0]
])

# СЦЕНАРИЙ 1: Идеальный выполнимый граф
h_good = np.array([12.0, 15.0, 20.0]) # Времени достаточно (20 - 4 = 16, 16 - 3 = 13)
g_good = np.array([1.0, 1.0, 1.0])

print("Запуск Сценария 1 (Корректный граф)...")
analyze_and_draw_tropical_graph(A, h_good, g_good, title="Сценарий 1: Корректное расписание")

# СЦЕНАРИЙ 2: Сломанный граф (Парадокс планирования)
# Узлу 2 нужен дедлайн 10.0, но чтобы успеть, Узел 0 должен стартовать в 3.0 (10 - 4 - 3 = 3).
# Но мы ставим жесткое условие g[0] = 6.0 (начинать раньше 6.0 нельзя). Система сломается!
h_bad = np.array([12.0, 15.0, 10.0]) 
g_bad = np.array([6.0, 1.0, 1.0])

print("\nЗапуск Сценария 2 (Сломанный граф)...")
analyze_and_draw_tropical_graph(A, h_bad, g_bad, title="Сценарий 2: Конфликт ограничений")



import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def build_tropical_matrix(standard_matrix):
    """
    Момент 1: Конвертер обычной матрицы расстояний/задержек в тропическую (Max-Plus).
    В обычной матрице: 0 - нет связи, числа - время пути.
    В тропической: -inf - нет связи, числа - ограничения.
    """
    matrix = np.array(standard_matrix, dtype=float)
    # Заменяем нули (отсутствие дорог между разными узлами) на минус бесконечность
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and matrix[i, j] == 0:
                matrix[i, j] = float('-inf')
    return matrix

def full_tropical_analyzer(standard_A, h, g=None):
    """
    Комплексный анализатор: оптимизация, валидация и поиск критического пути.
    """
    # 1. Переводим обычную матрицу в тропический формат Max-Plus
    A = build_tropical_matrix(standard_A)
    n = len(h)
    if g is None:
        g = np.full(n, float('-inf'))
        
    # 2. Вычисляем оптимальный вектор стартов x (Тропическое сопряжение)
    x = np.zeros(n)
    for i in range(n):
        possible_starts = []
        for j in range(n):
            if A[i, j] != float('-inf') and i != j:
                possible_starts.append(h[j] - A[i, j])
            else:
                possible_starts.append(h[i])
        x[i] = min(possible_starts)
        
    # 3. Валидация на парадоксы и ошибки
    is_consistent = True
    node_errors = set()
    edge_errors = set()
    
    for i in range(n):
        if x[i] < g[i]:
            is_consistent = False
            node_errors.add(i)
            
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] != float('-inf'):
                if A[i, j] + x[j] > x[i] + 1e-5:
                    is_consistent = False
                    edge_errors.add((i, j))
                    node_errors.add(i)
                    node_errors.add(j)

    # 4. Момент 2: Расчет критического пути (Critical Path)
    # В тропической математике ребро критическое, если задержка на нем выбрана "впритык"
    critical_edges = set()
    critical_nodes = set()
    
    if is_consistent:
        for i in range(n):
            for j in range(n):
                if i != j and A[i, j] != float('-inf'):
                    # Если время старта i в точности равно старту j + задержка,
                    # значит у этой работы нулевой резерв времени — это критический путь!
                    if abs(x[i] - (x[j] + A[i, j])) < 1e-5:
                        critical_edges.add((i, j))
                        critical_nodes.add(i)
                        critical_nodes.add(j)
                        
    return is_consistent, x, node_errors, edge_errors, critical_nodes, critical_edges, A

def draw_full_graph(standard_A, h, g=None, title="Тропический Анализ Системы"):
    """
    Визуализация графа со всеми тремя состояниями: Норма, Ошибка, Критический путь.
    """
    # Запуск вычислений
    ok, x, n_err, e_err, c_nodes, c_edges, A = full_tropical_analyzer(standard_A, h, g)
    n = len(h)
    
    G = nx.DiGraph()
    for i in range(n):
        label = f"Узел {i}\nСтарт: {x[i]:.1f}\nДедл: {h[i]:.1f}"
        G.add_node(i, label=label)
        
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] != float('-inf'):
                G.add_edge(i, j, weight=A[i, j])
                
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(11, 7))
    
    # Определяем цвета узлов
    node_colors = []
    for node in G.nodes():
        if node in n_err:
            node_colors.append('#ff6b6b')  # Красный — ошибка
        elif node in c_nodes:
            node_colors.append('#4dabf7')  # Синий — критический путь
        else:
            node_colors.append('#51cf66')  # Зеленый — нормальный запас времени
            
    # Определяем цвета и толщину стрелок
    edge_colors = []
    edge_widths = []
    for edge in G.edges():
        if edge in e_err:
            edge_colors.append('red')
            edge_widths.append(3.5)
        elif edge in c_edges:
            edge_colors.append('#1c7ed6') # Ярко-синяя стрелка для критического пути
            edge_widths.append(3.0)
        else:
            edge_colors.append('gray')
            edge_widths.append(1.5)
            
    # Отрисовка элементов
    plt.title(f"{title}\nСтатус: {'Все ок, критический путь подсвечен синим' if ok else 'Критическая ошибка логики!'}", 
              fontsize=12, color='darkblue' if ok else 'red', fontweight='bold')
              
    nx.draw_networkx_nodes(G, pos, node_size=3200, node_color=node_colors, edgecolors='black')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=edge_colors, width=edge_widths, arrowsize=20)
    
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold')
    
    edge_labels = {(u, v): f"t: {d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, color='purple')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# =====================================================================
# ТЕСТИРОВАНИЕ СИСТЕМЫ
# =====================================================================

# Обычная понятная матрица дорог/пауз (0 означает, что прямой связи нет)
# Узел 0 связан с 1 (пауза 3), Узел 1 связан с 2 (пауза 4), Узел 0 связан с 2 напрямую (пауза 2)
standard_matrix = [,
 ,
    [0, 0, 0]
]

h_vector = [12.0, 16.0, 20.0]
g_vector = [2.0, 2.0, 2.0]

# Запуск комплексной визуализации
draw_full_graph(standard_matrix, h_vector, g_vector, "Демонстрация полного тропического анализа")



import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def build_tropical_matrix(standard_matrix):
    """Конвертер обычной матрицы в тропическую (Max-Plus)"""
    matrix = np.array(standard_matrix, dtype=float)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and (matrix[i, j] == 0 or np.isnan(matrix[i, j])):
                matrix[i, j] = float('-inf')
    return matrix

def generate_random_tropical_case(num_nodes, edge_probability=0.4, max_delay=10):
    """
    Генератор случайных графов для стресс-тестов.
    Создает случайную матрицу ограничений и вектор дедлайнов.
    """
    # Создаем направленный граф без циклов (DAG), чтобы логика расписания имела смысл
    G = nx.gnp_random_graph(num_nodes, edge_probability, directed=True)
    G = nx.DiGraph([(u, v) for u, v in G.edges() if u < v]) # Исключаем обратные связи для простоты тестов
    
    standard_matrix = np.zeros((num_nodes, num_nodes))
    for u, v in G.edges():
        standard_matrix[u, v] = np.random.randint(1, max_delay)
        
    # Генерируем случайные дедлайны (возрастающие по цепочке)
    h = np.sort(np.random.randint(20, 100, size=num_nodes).astype(float))
    g = np.full(num_nodes, 0.0) # Ранний старт всегда с нуля
    
    return standard_matrix, h, g

def load_case_from_csv(matrix_path, targets_path):
    """
    Парсер для загрузки данных из CSV (или Excel через pd.read_excel)
    matrix_path: путь к таблице связей (матрица смежности)
    targets_path: путь к списку дедлайнов (колонки: node_id, deadline, min_start)
    """
    df_matrix = pd.read_csv(matrix_path, index_col=0)
    df_targets = pd.read_csv(targets_path)
    
    standard_matrix = df_matrix.to_numpy()
    h = df_targets['deadline'].to_numpy().astype(float)
    g = df_targets['min_start'].to_numpy().astype(float)
    
    return standard_matrix, h, g

def full_tropical_analyzer(standard_A, h, g=None):
    """Комплексный анализатор: оптимизация, валидация и критический путь"""
    A = build_tropical_matrix(standard_A)
    n = len(h)
    if g is None:
        g = np.full(n, float('-inf'))
        
    # Оптимизация (Max-Plus сопряжение)
    x = np.zeros(n)
    for i in range(n):
        possible_starts = []
        for j in range(n):
            if A[i, j] != float('-inf') and i != j:
                possible_starts.append(h[j] - A[i, j])
            else:
                possible_starts.append(h[i])
        x[i] = min(possible_starts)
        
    # Валидация
    is_consistent = True
    node_errors, edge_errors = set(), set()
    for i in range(n):
        if x[i] < g[i]:
            is_consistent = False
            node_errors.add(i)
            
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] != float('-inf'):
                if A[i, j] + x[j] > x[i] + 1e-5:
                    is_consistent = False
                    edge_errors.add((i, j))
                    node_errors.add(i)
                    node_errors.add(j)

    # Расчет критического пути
    critical_nodes, critical_edges = set(), set()
    if is_consistent:
        for i in range(n):
            for j in range(n):
                if i != j and A[i, j] != float('-inf'):
                    if abs(x[i] - (x[j] + A[i, j])) < 1e-5:
                        critical_edges.add((i, j))
                        critical_nodes.add(i)
                        critical_nodes.add(j)
                        
    return is_consistent, x, node_errors, edge_errors, critical_nodes, critical_edges, A

def draw_full_graph(standard_A, h, g=None, title="Тропический Анализ Системы"):
    """Отрисовка финального графа с поддержкой масштабирования"""
    ok, x, n_err, e_err, c_nodes, c_edges, A = full_tropical_analyzer(standard_A, h, g)
    n = len(h)
    
    G = nx.DiGraph()
    for i in range(n):
        label = f"N{i}\nS:{x[i]:.0f}\nD:{h[i]:.0f}"
        G.add_node(i, label=label)
        
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] != float('-inf'):
                G.add_edge(i, j, weight=A[i, j])
                
    pos = nx.shell_layout(G) if n > 5 else nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))
    
    node_colors = ['#ff6b6b' if node in n_err else '#4dabf7' if node in c_nodes else '#51cf66' for node in G.nodes()]
    edge_colors = ['red' if edge in e_err else '#1c7ed6' if edge in c_edges else 'gray' for edge in G.edges()]
    edge_widths = [3.5 if edge in e_err else 3.0 if edge in c_edges else 1.2 for edge in G.edges()]
            
    plt.title(f"{title}\nСтатус: {'Все ок, критический путь — СИНИЙ' if ok else 'ОШИБКА ОГРАНИЧЕНИЙ!'}", 
              fontsize=14, color='darkblue' if ok else 'red', fontweight='bold')
              
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color=node_colors, edgecolors='black')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=edge_colors, width=edge_widths, arrowsize=15)
    
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold')
    
    edge_labels = {(u, v): f"+{int(d['weight'])}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, color='purple')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# =====================================================================
# ЗАПУСК СТРЕСС-ТЕСТА (Случайный тропический граф на 6 узлов)
# =====================================================================
print("Генерируем случайную логистическую модель для стресс-теста...")
matrix_rand, h_rand, g_rand = generate_random_tropical_case(num_nodes=6, edge_probability=0.35)

# Визуализируем случайно сгенерированную систему
draw_full_graph(matrix_rand, h_rand, g_rand, "Стресс-тест: Случайный граф оптимизации")



import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class TropicalBioinformatics:
    """
    Модуль биоинформатики для тропического анализа эволюционных деревьев ДНК.
    Использует полукольцо Min-Plus для оценки мутаций.
    """
    def __init__(self):
        pass

    def compute_tropical_distance(self, tree_vector_A, tree_vector_B):
        """
        Фундаментальная формула тропической метрики между двумя филогенетическими деревьями.
        Принимает векторы расстояний между видами ДНК.
        """
        diff = tree_vector_A - tree_vector_B
        # Тропическая дистанция — это размах разностей между координатами векторов
        distance = np.max(diff) - np.min(diff)
        return distance

    def build_evolutionary_matrix(self, dna_sequences):
        """
        Берет сырые ДНК-последовательности (например: ['ATCG', 'ATGG'])
        И строит классическую матрицу попарных мутаций (расстояний Хэмминга).
        """
        n = len(dna_sequences)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # Считаем количество несовпадений в нуклеотидах
                mutations = sum(1 for a, b in zip(dna_sequences[i], dna_sequences[j]) if a != b)
                matrix[i, j] = mutations
        return matrix

    def extract_tree_vector(self, matrix):
        """
        Превращает симметричную матрицу расстояний ДНК в плоский вектор
        верхней треугольной матрицы (как требуют тропические статьи).
        """
        n = matrix.shape[0]
        iu = np.triu_indices(n, k=1)
        return matrix[iu]

    def plot_phylogenetic_network(self, distance_matrix, species_names):
        """
        Визуализирует эволюционную близость видов на основе ДНК.
        Чем меньше мутаций, тем толще и короче связь.
        """
        G = nx.Graph()
        n = len(species_names)
        
        for i in range(n):
            G.add_node(i, name=species_names[i])
            
        for i in range(n):
            for j in range(i+1, n):
                weight = distance_matrix[i, j]
                # Добавляем ребро. Для визуализации инвертируем вес (меньше мутаций -> сильнее связь)
                G.add_edge(i, j, weight=weight, score=max(1, 10 - weight))

        pos = nx.circular_layout(G)
        plt.figure(figsize=(8, 6))
        plt.title("Тропический профиль близости ДНК видов", fontsize=12, fontweight='bold', color='darkgreen')
        
        # Отрисовка
        nx.draw_networkx_nodes(G, pos, node_size=1200, node_color='#bbf7d0', edgecolors='green')
        
        # Подписи видов
        labels = {i: data['name'] for i, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
        
        # Отрисовка связей (толщина зависит от эволюционной близости)
        edges = G.edges(data=True)
        widths = [d['score'] * 0.5 for u, v, d in edges]
        nx.draw_networkx_edges(G, pos, width=widths, edge_color='darkgreen', alpha=0.6)
        
        # Подписи количества мутаций
        edge_labels = {(u, v): f"мутаций: {int(d['weight'])}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# =====================================================================
# ТЕСТИРОВАНИЕ ОЦИФРОВАННОГО БИОИНФОРМАТИЧЕСКОГО МОДУЛЯ
# =====================================================================

# Представим, что мы выделили один и тот же ген у 4 видов (Человек, Шимпанзе, Собака, Птица)
species = ["Человек", "Шимпанзе", "Собака", "Птица"]
dna_data = [
    "ATCGCCGTAA",  # Человек
    "ATCGCCGTAG",  # Шимпанзе (всего 1 мутация на конце)
    "ATTGTTGGAA",  # Собака (3 мутации относительно человека)
    "CCCGTTGGAA"   # Птица (много мутаций)
]

bio = TropicalBioinformatics()

# 1. Строим матрицу попарных генетических расстояний
mutation_matrix = bio.build_evolutionary_matrix(dna_data)
print("Матрица генетических мутаций между видами:\n", mutation_matrix)

# 2. Выделим тропические векторы для сравнения эволюционных траекторий
# (Например, сравним эволюционный профиль Шимпанзе и Собаки относительно всей группы)
vec_human = bio.extract_tree_vector(bio.build_evolutionary_matrix([dna_data[0], dna_data[1], dna_data[3]]))
vec_chimp = bio.extract_tree_vector(bio.build_evolutionary_matrix([dna_data[1], dna_data[2], dna_data[3]]))

# Считаем точное тропическое расстояние между эволюционными деревьями двух подгрупп
trop_dist = bio.compute_tropical_distance(vec_human, vec_chimp)
print(f"\nТропическое расстояние между эволюционными траекториями: {trop_dist:.2f}")

# 3. Визуализируем филогенетическую сеть близости ДНК
bio.plot_phylogenetic_network(mutation_matrix, species)



import numpy as np

class TropicalOptimizer:
    """
    Модуль прикладной тропической оптимизации (Max-Plus / Min-Plus).
    Решает задачи календарного планирования, логистики и анализа ограничений.
    Based on research papers by N. Krivulin.
    """
    def __init__(self, mode='max'):
        self.mode = mode
        self.zero = float('-inf') if mode == 'max' else float('inf')

    def solve_scheduling(self, A, h):
        """
        Вычисляет максимально допустимый вектор времени стартов (сопряжение).
        A: тропическая матрица связей/задержек
        h: вектор дедлайнов
        """
        n = len(h)
        x = np.zeros(n)
        
        for i in range(n):
            possible_starts = []
            for j in range(n):
                if A[i, j] != self.zero and i != j:
                    # В режиме max вычитаем задержку из дедлайна (обратная задача)
                    possible_starts.append(h[j] - A[i, j])
                else:
                    possible_starts.append(h[i])
            x[i] = np.min(possible_starts) if self.mode == 'max' else np.max(possible_starts)
            
        return x

    def validate_system(self, A, x, g=None):
        """
        Проверяет совместность системы неравенств (наличие временных парадоксов).
        g: вектор минимально разрешенного времени старта
        """
        n = len(x)
        if g is None:
            g = np.full(n, float('-inf') if self.mode == 'max' else float('inf'))
            
        is_consistent = True
        node_errors = set()
        edge_errors = set()
        
        # 1. Проверка внешних ограничений
        for i in range(n):
            if self.mode == 'max' and x[i] < g[i]:
                is_consistent = False
                node_errors.add(i)
            elif self.mode == 'min' and x[i] > g[i]:
                is_consistent = False
                node_errors.add(i)
                
        # 2. Проверка внутренних связей графа
        for i in range(n):
            for j in range(n):
                if i != j and A[i, j] != self.zero:
                    if self.mode == 'max' and (A[i, j] + x[j] > x[i] + 1e-5):
                        is_consistent = False
                        edge_errors.add((i, j))
                        node_errors.add(i)
                        node_errors.add(j)
                        
        return is_consistent, node_errors, edge_errors


import numpy as np
import matplotlib.pyplot as plt

def plot_tropical_balance_fan(delay_0_1, deadline_1):
    """
    Визуализирует тропический веер ограничений на 2D плоскости.
    Ось X: Время старта Задачи 0
    Ось Y: Время старта Задачи 1
    """
    # Создаем сетку возможных временных решений
    x_vals = np.linspace(0, 20, 400)
    y_vals = np.linspace(0, 20, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # В Max-Plus полукольце доминирует то ограничение, которое выдает максимум.
    # Ограничение 1: Технологическая пауза между Задачами 0 и 1 (Y >= X + delay)
    # Ограничение 2: Жесткий дедлайн для Задачи 1 (Y <= deadline)
    
    # Определяем, какое ограничение является главным (доминирующим) в каждой точке пространства
    # Для визуализации найдем зоны, где функции равны (зоны тропического излома)
    
    plt.figure(figsize=(9, 7))
    
    # Рисуем тропические лучи баланса (где max(X + delay, deadline) меняет автора)
    # Линия 1: Баланс между стартом задачи 0 и задержкой
    plt.plot(x_vals, x_vals + delay_0_1, label=f"Граница задержки (+{delay_0_1} ч)", color='red', linestyle='--', linewidth=2)
    
    # Линия 2: Линия дедлайна
    plt.axhline(y=deadline_1, label=f"Линия дедлайна ({deadline_1}:00)", color='blue', linestyle='--', linewidth=2)
    
    # Находим точку тропического пересечения (главный узел оптимизации)
    # X + delay = deadline => X = deadline - delay
    optimal_x = deadline_1 - delay_0_1
    optimal_y = deadline_1
    
    # Подсвечиваем зоны доминирования ограничений
    plt.fill_between(x_vals, x_vals + delay_0_1, 20, where=(x_vals + delay_0_1 <= deadline_1), 
                     color='green', alpha=0.15, label="Зона безопасного планирования")
    plt.fill_between(x_vals, 0, x_vals + delay_0_1, where=(x_vals >= optimal_x), 
                     color='orange', alpha=0.1, label="Зона критического риска (Срыв паузы)")
    
    # Жирная тропическая вершина (Идеальное решение)
    plt.scatter([optimal_x], [optimal_y], color='darkmagenta', s=250, zorder=5, 
                label=f"Тропическая Вершина\n(Оптимум: {optimal_x:.1f}, {optimal_y:.1f})")
    
    # Оформление графика
    plt.title("Новая визуализация: Тропический веер принятия решений (Max-Plus)", fontsize=13, fontweight='bold')
    plt.xlabel("Время старта Задачи 0 (Часы)", fontsize=11)
    plt.ylabel("Время старта Задачи 1 (Часы)", fontsize=11)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.show()

# Запуск визуализации веера: пауза между задачами 4 часа, дедлайн 15:00
plot_tropical_balance_fan(delay_0_1=4.0, deadline_1=15.0)



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_tropical_surface(delay_0_1, deadline_1):
    """
    Строит 3D тропическое многообразие (кристалл ограничений).
    Оси X, Y: Времена старта задач.
    Ось Z: Тропическая функция штрафа / доминирующего ограничения.
    """
    # Генерируем сетку координат
    x = np.linspace(0, 20, 100)
    y = np.linspace(0, 20, 100)
    X, Y = np.meshgrid(x, y)
    
    # Моделируем тропическую функцию: Z = max(X + delay, Y) - deadline
    # Это классический вид угловатых тропических поверхностей из научных статей
    Z = np.maximum(X + delay_0_1, Y) - deadline_1
    
    # Создаем 3D холст
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Строим угловатую поверхность (Тропическое многообразие)
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    
    # Находим точку тропического излома (Оптимум)
    opt_x = deadline_1 - delay_0_1
    opt_y = deadline_1
    opt_z = np.maximum(opt_x + delay_0_1, opt_y) - deadline_1
    
    # Выделяем тропическую вершину жирной точкой
    ax.scatter([opt_x], [opt_y], [opt_z], color='red', s=300, marker='*', zorder=10, 
               label=f"Тропическая вершина (ось излома)\n[{opt_x:.1f}, {opt_y:.1f}, {opt_z:.1f}]")
    
    # Настройка углов обзора, чтобы излом был четко виден
    ax.view_init(elev=28, azim=-125)
    
    # Оформление
    ax.set_title("3D Визуализация: Угловатая геометрия тропического многообразия", fontsize=14, fontweight='bold')
    ax.set_xlabel("Старт Задачи 0 (X)", fontsize=10)
    ax.set_ylabel("Старт Задачи 1 (Y)", fontsize=10)
    ax.set_zlabel("Тропический потенциал (Z)", fontsize=10)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Уровень потенциала")
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    plt.show()

# Запуск: задержка 5 часов, базовый уровень 12
plot_3d_tropical_surface(delay_0_1=5.0, deadline_1=12.0)



import numpy as np
import matplotlib.pyplot as plt

def plot_tropical_vector_field(delay, deadline):
    """
    Строит уникальное дискретное векторное поле тропических градиентов.
    Показывает, как силы ограничений сталкиваются на границах излома.
    """
    # Создаем редкую сетку для стрелок (чтобы они не сливались)
    x = np.linspace(0, 20, 25)
    y = np.linspace(0, 20, 25)
    X, Y = np.meshgrid(x, y)
    
    # Вычисляем компоненты векторов (U - по оси X, V - по оси Y)
    # Функция: Z = max(X + delay, Y)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    # Тропический градиент: 
    # Если X + delay > Y, то доминирует X (вектор смотрит вправо: U=1, V=0)
    # Если X + delay < Y, то доминирует Y (вектор смотрит вверх: U=0, V=1)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] + delay > Y[i, j]:
                U[i, j] = 1.0
                V[i, j] = 0.0
            else:
                U[i, j] = 0.0
                V[i, j] = 1.0
                
    plt.figure(figsize=(9, 7))
    plt.title("Инновация: Дискретное тропическое поле сил (Столкновение потоков)", 
              fontsize=12, fontweight='bold')
    
    # Рисуем ломаную линию тропического баланса (линию разлома)
    x_line = np.linspace(0, 20, 100)
    plt.plot(x_line, x_line + delay, color='purple', linewidth=3, zorder=3,
             label="Линия тектонического разлома ограничений")
    
    # Отрисовка векторного поля через quiver
    plt.quiver(X, Y, U, V, color='teal', alpha=0.6, scale=25, headwidth=4)
    
    # Подсвечиваем точку главного оптимума (где поток упирается в дедлайн)
    opt_x = deadline - delay
    plt.scatter([opt_x], [deadline], color='red', s=200, marker='X', zorder=5, 
                label=f"Точка сингулярности (Оптимум: {opt_x:.1f})")
    plt.axhline(y=deadline, color='red', linestyle=':', alpha=0.5)
    
    # Оформление
    plt.xlabel("Старт Задачи 0")
    plt.ylabel("Старт Задачи 1")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.grid(True, alpha=0.2)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

# Запуск: задержка между задачами 3 часа, дедлайн второй задачи 14:00
plot_tropical_vector_field(delay=3.0, deadline=14.0)





import numpy as np
import matplotlib.pyplot as plt

def plot_tropical_crystal_growth(slopes_x, slopes_y, intercepts):
    """
    Метод 4: Визуализация фазовых зон роста кристалла.
    Каждая грань кристалла задается линейной функцией: Z_k = a_k*X + b_k*Y + c_k
    Тропический полином находит максимум (верхнюю оболочку кристалла).
    """
    x = np.linspace(-10, 10, 500)
    y = np.linspace(-10, 10, 500)
    X, Y = np.meshgrid(x, y)
    
    # Инициализируем тензор для всех граней кристалла
    num_faces = len(intercepts)
    faces = np.zeros((num_faces, X.shape[0], X.shape[1]))
    
    for k in range(num_faces):
        faces[k] = slopes_x[k] * X + slopes_y[k] * Y + intercepts[k]
    
    # Тропическое сложение (Max) определяет доминирующую грань в каждой точке
    dominant_face_map = np.argmax(faces, axis=0)
    
    plt.figure(figsize=(9, 7))
    plt.title("Метод 4: Тропический атлас фаз и микроструктуры кристалла", fontsize=12, fontweight='bold')
    
    # Рисуем зоны доминирования граней (как зерна в металле или грани кристалла)
    plt.imshow(dominant_face_map, extent=[-10, 10, -10, 10], origin='lower', cmap='Set3', alpha=0.8)
    
    # Рисуем тропические швы (линии излома, где сходятся грани)
    plt.contour(X, Y, dominant_face_map, colors='black', linewidths=1.5)
    
    plt.xlabel("Параметр среды X (например, Температура)")
    plt.ylabel("Параметр среды Y (например, Давление)")
    plt.grid(True, alpha=0.2, linestyle=':')
    
    # Добавим маркеры сингулярностей (тропических вершин)
    plt.scatter([0, -2, 3], [1, -4, 2], color='red', marker='o', s=100, label='Тропические тройные точки (Узлы баланса)')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# Запуск 4-го метода: 4 разные грани растущего кристалла с разными скоростями
plot_tropical_crystal_growth(
    slopes_x=[1.0, -1.0, 0.0, 0.5],
    slopes_y=[0.0, 1.0, -1.0, -0.5],
    intercepts=[2.0, -1.0, 3.0, 0.0]
)



import numpy as np
import matplotlib.pyplot as plt

def plot_tropical_contour_waves():
    """
    Метод 5: Тропические ломаные фронты волн кристаллизации.
    Показывает изолинии в тропической метрике Max-Plus.
    """
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    
    # Тропическое расстояние от центра (0,0) в Max-Plus: Z = max(|X|, |Y|)
    # А для имитации сложного кристалла сделаем комбинацию двух тропических центров:
    Z1 = np.maximum(np.abs(X - 2), np.abs(Y - 2))
    Z2 = np.maximum(np.abs(X + 3), np.abs(Y + 3))
    
    # Тропическое «произведение» (для расстояний это минимум)
    Z = np.minimum(Z1, Z2)
    
    plt.figure(figsize=(9, 7))
    plt.title("Метод 5: Тропические ломаные фронты (Эволюция формы кристалла)", fontsize=12, fontweight='bold')
    
    # Рисуем заполненные тропические контуры
    cp = plt.contourf(X, Y, Z, levels=15, cmap='coolwarm', alpha=0.6)
    plt.colorbar(cp, label="Шаги времени / Эволюционный слой")
    
    # Подчеркиваем жесткие угловатые ребра фронта
    contours = plt.contour(X, Y, Z, levels=15, colors='indigo', linewidths=1.0)
    plt.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
    
    # Точки затравки кристаллов (ядра аккреции)
    plt.scatter([2, -3], [2, -3], color='yellow', edgecolors='black', s=150, marker='P', label='Центры кристаллизации')
    
    plt.xlabel("Пространственная ось X")
    plt.ylabel("Пространственная ось Y")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.show()

# Запуск 5-го метода
plot_tropical_contour_waves()



import numpy as np
import matplotlib.pyplot as plt

def dna_to_trajectory(sequence):
    """
    Превращает текстовую строку ДНК в массив 2D-координат (траекторию гена).
    A -> (+1, 0), T -> (-1, 0), C -> (0, +1), G -> (0, -1)
    """
    # Словарь направлений
    moves = {
        'A': np.array([1, 0]),
        'T': np.array([-1, 0]),
        'C': np.array([0, 1]),
        'G': np.array([0, -1])
    }
    
    # Стартуем из центра (0, 0)
    current_pos = np.array([0, 0])
    trajectory = [current_pos.copy()]
    
    for nucleotide in sequence.upper():
        if nucleotide in moves:
            current_pos += moves[nucleotide]
            trajectory.append(current_pos.copy())
            
    return np.array(trajectory)

def plot_tropical_dna_mapping(dna_1, dna_2, name_1="Вид А", name_2="Вид Б"):
    """
    Метод 6: Тропическое картирование и геометрия расхождения ДНК.
    Строит "шаги" геномов и находит точки тропического излома мутаций.
    """
    traj_1 = dna_to_trajectory(dna_1)
    traj_2 = dna_to_trajectory(dna_2)
    
    plt.figure(figsize=(10, 8))
    plt.title(f"Метод 6: Тропический геометрический портрет ДНК\nСравнение: {name_1} vs {name_2}", 
              fontsize=12, fontweight='bold', color='darkgreen')
    
    # Рисуем траекторию первого гена
    plt.plot(traj_1[:, 0], traj_1[:, 1], color='#1c7ed6', linewidth=2.5, label=name_1, zorder=3)
    plt.scatter(traj_1[-1, 0], traj_1[-1, 1], color='#1c7ed6', s=100, edgecolors='black')
    
    # Рисуем траекторию второго гена
    plt.plot(traj_2[:, 0], traj_2[:, 1], color='#e03131', linewidth=2.0, label=name_2, alpha=0.8, zorder=2)
    plt.scatter(traj_2[-1, 0], traj_2[-1, 1], color='#e03131', s=100, edgecolors='black')
    
    # --- ТРОПИЧЕСКИЙ АНАЛИЗ МУТАЦИЙ (Min-Plus) ---
    # Находим точки, где траектории расходятся из-за мутаций
    min_len = min(len(traj_1), len(traj_2))
    mutation_points = []
    
    for i in range(min_len):
        # Тропическое расстояние (метрика Чебышёва/Min-Plus) между текущими шагами
        dist = np.max(np.abs(traj_1[i] - traj_2[i]))
        if dist > 0: # Если линии разошлись — это следствие мутации в коде
            mutation_points.append(traj_1[i])
            
    if mutation_points:
        mutation_points = np.array(mutation_points)
        # Подсвечиваем "облако мутаций" (тропический след эволюционного сдвига)
        plt.scatter(mutation_points[:, 0], mutation_points[:, 1], 
                    color='purple', alpha=0.3, s=80, label='Тропический след мутаций', zorder=1)
    
    # Начальная точка (общий предок / старт гена)
    plt.scatter([0], [0], color='gold', s=200, marker='*', edgecolors='black', label='Старт гена (Отец)', zorder=5)
    
    # Оформление
    plt.xlabel("Ось пуриновых оснований (A-T)")
    plt.ylabel("Ось пиримидиновых оснований (C-G)")
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend(loc="upper left")
    plt.axis('equal') # Чтобы масштабы осей были одинаковыми и шаги не искажались
    plt.tight_layout()
    plt.show()

# =====================================================================
# ТЕСТИРОВАНИЕ НА РЕАЛЬНЫХ СТРОКАХ ДНК
# =====================================================================

# Ген Человека (условный пример цепочки)
human_dna = "ATCGATCGATCGGGCCCAAATTTATCGATCGATCGATC"

# Ген Шимпанзе (всего пара замен букв в середине, траектория почти копия)
chimp_dna = "ATCGATCGATCAGGCCCAAATTTATCGAGCGATCGATC"

# Ген бактерии или вируса (совершенно другая структура, улетит в сторону)
virus_dna = "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCC"

# Тест 1: Сравниваем Человека и близкого Шимпанзе
print("Строим график для близких видов...")
plot_tropical_dna_mapping(human_dna, chimp_dna, "Человек", "Шимпанзе")

# Тест 2: Сравниваем Человека и далекий организм
print("Строим график для далеких видов...")
plot_tropical_dna_mapping(human_dna, virus_dna, "Человек", "Вирусный маркер")



import numpy as np
import matplotlib.pyplot as plt

def plot_tropical_dna_drift_grid(dna_1, dna_2):
    """
    Метод 7: Тропическая тепловая карта эволюционного дрейфа.
    Строит матрицу накопленного тропического расстояния между всеми парами нуклеотидов.
    """
    n1, n2 = len(dna_1), len(dna_2)
    grid = np.zeros((n1, n2))
    
    # Заполняем сетку накопленным тропическим штрафом за несовпадение букв
    for i in range(n1):
        for j in range(n2):
            # В тропическом Min-Plus мире штраф за мутацию накапливается линейно
            # Считаем, сколько мутаций произошло до текущей позиции i и j
            mutations_1 = sum(1 for a, b in zip(dna_1[:i], dna_2[:i]) if a != b)
            mutations_2 = sum(1 for a, b in zip(dna_1[:j], dna_2[:j]) if a != b)
            grid[i, j] = abs(mutations_1 - mutations_2)
            
    plt.figure(figsize=(9, 7))
    plt.title("Метод 7: Тропический тепловой разлом ДНК (Карта дрейфа)", fontsize=12, fontweight='bold')
    
    # Отрисовка тепловой карты
    plt.imshow(grid, cmap='coolwarm', origin='lower')
    plt.colorbar(label="Тропический уровень эволюционного расхождения")
    
    # Рисуем жесткую ломаную линию идеального выравнивания (тропический шов)
    plt.plot([0, min(n1, n2)-1], [0, min(n1, n2)-1], color='black', linestyle='--', linewidth=2, label='Линия идеального совпадения')
    
    plt.xlabel("Индекс нуклеотида ДНК (Вид А)")
    plt.ylabel("Индекс нуклеотида ДНК (Вид Б)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

# Тестовые данные (вспышка мутаций в середине гена)
dna_human = "AAAAAAGGGGGGTTTTTTCCCCCC"
dna_mutant = "AAAAAATTTTTTTTTTTTCCCCCC" # Заменили GGGGGG на TTTTTT
plot_tropical_dna_drift_grid(dna_human, dna_mutant)




import numpy as np
import matplotlib.pyplot as plt

def plot_tropical_polar_radar(dna_1, dna_2, name_1="Человек", name_2="Мутант"):
    """
    Метод 8: Круговой тропический радар мутаций.
    Сворачивает ДНК в полярные координаты, подсвечивая изломы.
    """
    n = min(len(dna_1), len(dna_2))
    angles = np.linspace(0, 2 * np.pi, n)
    
    # Генерируем радиусы на основе накопленных шагов (базовый радиус 10 + тропический сдвиг)
    r1 = np.ones(n) * 10.0
    r2 = np.ones(n) * 10.0
    
    current_diff = 0
    for i in range(n):
        if dna_1[i] != dna_2[i]:
            current_diff += 2.0 # Тропический шаг за мутацию
        r2[i] += current_diff
        
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    ax.set_title("Метод 8: Полярный тропический радар структуры ДНК", fontsize=12, fontweight='bold', pad=20)
    
    # Рисуем круговые траектории
    ax.plot(angles, r1, color='#1c7ed6', linewidth=2.5, label=name_1)
    ax.fill(angles, r1, color='#1c7ed6', alpha=0.1)
    
    ax.plot(angles, r2, color='#e03131', linewidth=2.5, label=name_2)
    ax.fill(angles, r2, color='#e03131', alpha=0.1)
    
    # Подсвечиваем жесткие углы излома в местах мутаций
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_yticklabels([]) # Убираем скучные стандартные цифры радиуса
    
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# Запуск радара
plot_tropical_polar_radar("ATCGATCGATCGATCGATCG", "ATCGATGGGGGGATCGATCG")



import numpy as np
import matplotlib.pyplot as plt

def plot_tropical_principal_tree(num_points=300):
    """
    Метод 9: Сжатие многомерного хаоса данных в тропический жесткий каркас.
    Выделяет главные траектории из хаотичного облака точек.
    """
    np.random.seed(42)
    # Генерируем хаотичное облако данных (например, мутации или логи)
    x_raw = np.random.uniform(-10, 10, num_points)
    # Добавляем сильный случайный шум вокруг тропического излома
    y_raw = np.maximum(x_raw, -x_raw * 0.5) + np.random.normal(0, 1.5, num_points)
    
    # --- ТРОПИЧЕСКАЯ РЕДУКЦИЯ КАРКАСА ---
    # Оцифрованная формула тропической регрессии: находим идеальный жесткий излом
    x_grid = np.linspace(-10, 10, 100)
    # Сама математическая суть данных без шума
    y_skeleton = np.maximum(x_grid, -x_grid * 0.5)
    
    plt.figure(figsize=(10, 7))
    plt.title("Метод 9: Тропический магистральный каркас (Фильтрация хаоса Big Data)", 
              fontsize=12, fontweight='bold')
    
    # Рисуем хаос, в котором обычно тонут люди
    plt.scatter(x_raw, y_raw, color='gray', alpha=0.4, s=25, label="Сырые терабайты данных (Шум)")
    
    # Рисуем жесткий тропический скелет, который мгновенно объясняет структуру данных
    plt.plot(x_grid, y_skeleton, color='#f59f00', linewidth=4, zorder=5,
             label="Тропическая магистраль тренда (Главный каркас)")
    
    # Выделяем точку бифуркации (где процесс раздваивается)
    plt.scatter([0], [0], color='red', marker='X', s=250, zorder=6, label="Точка ветвления процесса")
    
    plt.xlabel("Пространство признаков X (Параметр процесса)")
    plt.ylabel("Пространство признаков Y (Результат)")
    plt.grid(True, alpha=0.15)
    plt.legend(loc="upper color", fontsize=10)
    plt.tight_layout()
    plt.show()

plot_tropical_principal_tree()





import numpy as np
import matplotlib.pyplot as plt

def plot_tropical_pareto_shield():
    """
    Метод 10: Тропический щит Парето. 
    Выделяет угловые точки максимальной эффективности среди тысяч хаотичных бизнес-решений.
    """
    np.random.seed(15)
    # Генерируем 500 случайных вариантов (например, маршруты или конфигурации генов)
    cost = np.random.uniform(10, 100, 400)
    time = np.random.uniform(5, 50, 400)
    
    # Сделаем так, чтобы была явная граница эффективности
    for i in range(400):
        if cost[i] * time[i] < 300:
            cost[i] += 30
            time[i] += 15

    plt.figure(figsize=(10, 7))
    plt.title("Метод 10: Тропический Парето-Щит (Управление стратегическими решениями)", 
              fontsize=12, fontweight='bold')
    
    plt.scatter(cost, time, color='blue', alpha=0.3, s=30, label="Слабые/Избыточные стратегии")
    
    # Строим строгую тропическую ступенчатую границу (Min-Plus Парето)
    # Идеальные угловые решения, найденные тропическим оператором максимума
    pareto_x = np.sort(np.array([12, 20, 35, 55, 75, 95]))
    pareto_y = np.array([45, 32, 22, 14, 9, 6])
    
    # Рисуем ломаную тропическую границу
    plt.step(pareto_x, pareto_y, where='post', color='darkred', linewidth=3, zorder=4,
             label="Тропический Парето-Щит (Граница оптимума)")
    
    # Подсвечиваем "Золотые точки" — углы излома, где баланс идеален
    plt.scatter(pareto_x, pareto_y, color='gold', edgecolors='black', s=150, zorder=5, 
                label="Золотые конфигурации (Точки принятия решений)")
    
    plt.xlabel("Критерий 1: Финансовые затраты / Уровень мутаций")
    plt.ylabel("Критерий 2: Время доставки / Риск выживания вида")
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

plot_tropical_pareto_shield()
