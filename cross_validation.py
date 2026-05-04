import numpy as np
from sklearn.model_selection import KFold
from perceptron import Perceptron

# Бонус - кросс-валидация
def run_cross_validation(X, y, k=5):
    print(f"\n--- Запуск {k}-Fold Кросс-валидации ---")
    print("Гиперпараметры: epochs=50, lr=0.1, batch_size=32, gamma=0.9")
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        # 1. Получаем срезы данных
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 2. Нормализуем
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        
        # 3. Создаем модель
        model = Perceptron(gamma=0.9)
        
        # 4. Обучаем
        model.fit(X_train, y_train, epochs=50, lr=0.1, batch_size=32)
        
        # 5. Оцениваем
        test_preds = model.predict(X_test)
        acc = np.mean(test_preds == y_test) * 100
        
        # 6. Сохраняем результат
        print(f"Фолд {fold + 1}: Точность = {acc:.2f}%")
        fold_accuracies.append(acc)
        
    # 7. Средний результат
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"Средняя точность кросс-валидации: {mean_acc:.2f}% (±{std_acc:.2f}%)\n")
    return mean_acc