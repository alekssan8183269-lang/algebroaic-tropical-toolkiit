import numpy as np

class TropicalRegressor:
    """
    Тропический классификатор/регрессор (Max-Plus Regression).
    Аппроксимирует данные с помощью тропического полинома:
    F(x) = max(w_1 + x_1, w_2 + x_2, ..., w_d + x_d)
    Подходит для построения угловатых разделяющих границ в ML.
    """
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        """
        Обучение модели (оценка весов w).
        По статьям тропического ML, оптимальные веса для Max-Plus 
        полукольца вычисляются аналитически через тропическое сопряжение.
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Находим веса как нижнюю оценку разностей: w_j = min_i (y_i - x_ij)
        # Это гарантирует, что тропический полином будет огибать данные снизу
        num_features = X.shape[1]
        self.weights = np.zeros(num_features)
        
        for j in range(num_features):
            self.weights[j] = np.min(y - X[:, j])
            
        return self

    def predict(self, X):
        """Предсказание целевой переменной с использованием только операции сложения и max."""
        X = np.array(X, dtype=float)
        if self.weights is None:
            raise ValueError("Модель еще не обучена! Сначала вызовите метод fit().")
            
        # Вычисляем тропический полином: max(X + weights)
        return np.max(X + self.weights, axis=1)
