import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TropicalVisualizer:
    """Глобальный визуализатор TroPy: 10 методов отображения тропической математики."""
    
    @staticmethod
    def method_1_network_graph(A, x, h, n_err, e_err, c_nodes, c_edges, title="M1: Граф Сети"):
        """Метод 1: Сетевые графы связей, ошибок и критических путей."""
        n = len(h)
        G = nx.DiGraph()
        for i in range(n): G.add_node(i, label=f"N{i}\nS:{x[i]:.1f}")
        for i in range(n):
            for j in range(n):
                if i != j and A[i, j] != float('-inf') and A[i, j] != float('inf'):
                    G.add_edge(i, j, weight=A[i, j])
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(6, 4))
        node_colors = ['#ff6b6b' if n in n_err else '#4dabf7' if n in c_nodes else '#51cf66' for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=node_colors, edgecolors='black')
        nx.draw_networkx_edges(G, pos, width=2, edge_color='gray')
        nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=8)
        plt.title(title)
        plt.axis('off')
        plt.show()

    @staticmethod
    def method_2_balance_fan(delay, deadline):
        """Метод 2: 2D-веера доминирования ограничений."""
        x = np.linspace(0, 20, 100)
        plt.figure(figsize=(6, 4))
        plt.plot(x, x + delay, color='red', linestyle='--', label="Пауза")
        plt.axhline(y=deadline, color='blue', linestyle='--', label="Дедлайн")
        plt.fill_between(x, x + delay, 20, where=(x + delay <= deadline), color='green', alpha=0.1)
        plt.scatter([deadline - delay], [deadline], color='purple', s=150, zorder=5, label="Вершина")
        plt.title("M2: 2D Веер Баланса")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def method_3_crystal_surface(delay, deadline):
        """Метод 3: 3D-кристаллы штрафных поверхностей."""
        x = np.linspace(0, 20, 50)
        y = np.linspace(0, 20, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.maximum(X + delay, Y) - deadline
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        ax.set_title("M3: 3D Кристалл Рельефа")
        plt.show()

    @staticmethod
    def method_4_gradient_field(delay):
        """Метод 4: Дискретные векторные поля сил."""
        x = np.linspace(0, 20, 15)
        y = np.linspace(0, 20, 15)
        X, Y = np.meshgrid(x, y)
        U = np.where(X + delay > Y, 1.0, 0.0)
        V = np.where(X + delay > Y, 0.0, 1.0)
        plt.figure(figsize=(6, 4))
        plt.quiver(X, Y, U, V, color='teal', alpha=0.6)
        plt.plot(np.linspace(0,20,100), np.linspace(0,20,100)+delay, color='purple')
        plt.title("M4: Поле Градиентов Стыка")
        plt.show()

    @staticmethod
    def method_5_contour_waves():
        """Метод 5: Ломаные тропические фронты волн."""
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.minimum(np.maximum(np.abs(X-2), np.abs(Y-2)), np.maximum(np.abs(X+3), np.abs(Y+3)))
        plt.figure(figsize=(6, 4))
        plt.contourf(X, Y, Z, levels=10, cmap='coolwarm', alpha=0.5)
        plt.contour(X, Y, Z, levels=10, colors='indigo')
        plt.title("M5: Ломаные Фронты Контуров")
        plt.show()

    @staticmethod
    def method_6_dna_trajectory(traj_1, traj_2, name_1="A", name_2="B", mut_points=None):
        """Метод 6: Геометрические 2D траектории цепочек ДНК."""
        plt.figure(figsize=(6, 4))
        plt.plot(traj_1[:, 0], traj_1[:, 1], color='#1c7ed6', linewidth=2, label=name_1)
        plt.plot(traj_2[:, 0], traj_2[:, 1], color='#e03131', linewidth=1.5, label=name_2)
        if mut_points is not None and len(mut_points) > 0:
            plt.scatter(mut_points[:, 0], mut_points[:, 1], color='purple', alpha=0.4, label='Мутации')
        plt.scatter([0], [0], color='gold', marker='*', s=100, label='Предок')
        plt.title("M6: Геометрический Портрет ДНК")
        plt.legend()
        plt.axis('equal')
        plt.show()

    @staticmethod
    def method_7_crystal_growth(slopes_x, slopes_y, intercepts):
        """Метод 7: Фазовый атлас зёрен растущего кристалл."""
        x = np.linspace(-5, 5, 200)
        y = np.linspace(-5, 5, 200)
        X, Y = np.meshgrid(x, y)
        faces = np.zeros((len(intercepts), X.shape[0], X.shape[1]))
        for k in range(len(intercepts)):
            faces[k] = slopes_x[k]*X + slopes_y[k]*Y + intercepts[k]
        dominant = np.argmax(faces, axis=0)
        plt.figure(figsize=(6, 4))
        plt.imshow(dominant, extent=[-5, 5, -5, 5], origin='lower', cmap='Set3')
        plt.contour(X, Y, dominant, colors='black', linewidths=0.8)
        plt.title("M7: Карта Фазовых Зёрен Кристалла")
        plt.show()

    @staticmethod
    def method_8_dna_drift_grid(grid):
        """Метод 8: Тепловые карты разломов эволюционного дрейфа."""
        plt.figure(figsize=(6, 4))
        plt.imshow(grid, cmap='coolwarm', origin='lower')
        plt.colorbar()
        plt.title("M8: Тепловой Разлом Дрейфа ДНК")
        plt.show()

    @staticmethod
    def method_9_polar_radar(angles, r1, r2, name_1="A", name_2="B"):
        """Метод 9: Полярные радары радиальных мутаций."""
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})
        ax.plot(angles, r1, color='#1c7ed6', label=name_1)
        ax.plot(angles, r2, color='#e03131', label=name_2)
        ax.fill(angles, r1, color='#1c7ed6', alpha=0.05)
        ax.fill(angles, r2, color='#e03131', alpha=0.05)
        plt.title("M9: Полярный Радар Мутаций ДНК", pad=15)
        plt.legend()
        plt.show()

    @staticmethod
    def method_10_pareto_shield(cost, time, px, py):
        """Метод 10: Ступенчатые тропические щиты границ Парето."""
        plt.figure(figsize=(6, 4))
        plt.scatter(cost, time, color='blue', alpha=0.3, label="Варианты")
        plt.step(px, py, where='post', color='darkred', linewidth=2.5, label="Парето-Щит")
        plt.scatter(px, py, color='gold', edgecolors='black', s=80, zorder=5)
        plt.title("M10: Тропический Парето-Щит")
        plt.legend()
        plt.show()
