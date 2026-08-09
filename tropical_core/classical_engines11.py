import numpy as np
import matplotlib.pyplot as plt

def plot_maslov_dequantization(steps=5):
    """
    Метод 11: Квантование Маслова в динамике.
    Показывает, как гладкое дифференциальное уравнение (физика)
    превращается в жесткую ломаную линию (Тропическая математика) при h -> 0.
    """
    x = np.linspace(-5, 5, 500)
    
    plt.figure(figsize=(10, 6))
    plt.title("Метод 11: Превращение гладкой физики в тропический излом (h -> 0)", 
              fontsize=12, fontweight='bold')
    
    # Значения параметра деквантования (аналог температуры или постоянной Планка)
    # От большого (плавный мир) до микроскопического (тропический мир)
    h_values = [2.0, 1.0, 0.5, 0.1, 0.01]
    colors = ['#ced4da', '#adb5bd', '#495057', '#1c7ed6', '#d9480f']
    
    for h, color in zip(h_values, colors):
        # Формула Маслова: h * ln( exp(x/h) + exp(0/h) )
        # При h -> 0 эта функция строго превращается в тропическое сложение: max(x, 0)
        with np.errstate(over='ignore'): # Защита от переполнения float при малых h
            smooth_curve = h * np.log(np.exp(x / h) + 1.0)
            
        label = f"Мир физики (h = {h})" if h > 0.1 else f"Тропический предел (h = {h})"
        plt.plot(x, smooth_curve, color=color, linewidth=2.5 if h == 0.01 else 1.5, label=label)

    # Идеальный тропический скелет Max-Plus для сравнения
    plt.plot(x, np.maximum(x, 0), color='black', linestyle=':', linewidth=2, label="Чистый Max-Plus закон")

    plt.xlabel("Пространственная координата (X)")
    plt.ylabel("Потенциал системы (Z)")
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()

# Запустим и посмотрим на это чудо физики
plot_maslov_dequantization()
