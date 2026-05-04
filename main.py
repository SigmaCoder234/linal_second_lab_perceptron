import numpy as np
import matplotlib.pyplot as plt
from data_utils import get_prepared_data
from data_utils import generate_custom_data
from sklearn.model_selection import train_test_split
from perceptron import Perceptron
from metrics import print_advanced_metrics, plot_roc_curve
from experiments import run_experiments
from cross_validation import run_cross_validation

def main():
    # 0. Выбор генератора данных
    print("""
Выберите тип генератора данных:
1 - классический датасет
2 - кастомный датасет
0 - Выход
        """)

    while True:
        data_choice = input("Ваш выбор: ")
        if data_choice in ['1', '2', '0']:
            break
        print("Ошибка! Пожалуйста, введите 1 или 2.")

    # 1. Получаем подготовленные данные
    if data_choice == '1':
        data = get_prepared_data()
        X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]

        # Собираем данные обратно для кросс-валидации
        X = np.vstack((X_train, X_test))
        y = np.hstack((y_train, y_test))
    elif data_choice == '2':
        print("""
Выберите тип кастомного датасета:
1 - linear
2 - xor
3 - circle
        """)
    
    elif data_choice == '0':
        exit()

        while True:
            data_type = input("Ваш выбор: ")
            if data_type in ['1', '2', '3']:
                break
            print("Ошибка! Пожалуйста, введите 1, 2 или 3.")
        
        if data_type == '1':
            data_type = 'linear'
        elif data_type == '2':
            data_type = 'xor'
        elif data_type == '3':
            data_type = 'circle'
        
        X, y = generate_custom_data(type=f'{data_type}', n_samples=500, noise=0.05)
    
        # Разбиваем на train и test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        
        # Нормализуем (считаем линейку по train, применяем к обоим)
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        
    # === Запускаем кросс-валидацию ===
    run_cross_validation(X, y, k=5)
    
    # 2. Создаем нейросеть
    model = Perceptron()
    
    # 3. Обучаем её
    print("Начинаем обучение...")
    train_losses, val_losses = model.fit(
        X=X_train, 
        y=y_train, 
        X_val=X_test,   # Передаем тест как валидацию
        y_val=y_test, 
        epochs=100, 
        lr=0.1, 
        batch_size=32
    )
    print("Обучение завершено!")
    
    # 4. Рисуем график падения ошибки
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation (Test) Loss', color='orange')
    plt.title('График функции потерь (Loss) от эпохи')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss (Кросс-энтропия)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 5. Оценка точности (Accuracy)
    # Получаем предсказания (нули и единицы)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Считаем долю совпадений
    train_acc = np.mean(train_preds == y_train) * 100
    test_acc = np.mean(test_preds == y_test) * 100
    
    print(f"Точность на обучении: {train_acc:.2f}%")
    print(f"Точность на тесте: {test_acc:.2f}%")

    # 6. Визуализация разделяющей прямой
    plt.figure(figsize=(10, 6))
    
    # Рисуем все точки (c=y_train задает цвет в зависимости от класса)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.7, edgecolors='k')
    
    # Рисуем прямую w0*x0 + w1*x1 + b = 0  =>  x1 = -(w0*x0 + b) / w1
    x0_min, x0_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    x0_line = np.array([x0_min, x0_max])
    x1_line = -(model.w[0] * x0_line + model.b) / model.w[1]
    
    plt.plot(x0_line, x1_line, color='green', linewidth=3, label='Decision Boundary')
    
    plt.title('Разделяющая прямая перцептрона')
    plt.xlabel('Признак 1 (Нормализованный)')
    plt.ylabel('Признак 2 (Нормализованный)')
    plt.legend()
    plt.grid(True)
    plt.show()

    run_experiments(X_train, y_train, X_test, y_test)

    print_advanced_metrics(y_test, test_preds, model.forward(X_test))
    plot_roc_curve(y_test, model.forward(X_test))

if __name__ == '__main__':
    main()