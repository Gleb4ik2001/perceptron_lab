import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

# Загрузка данных
with open('mnist.pkl', 'rb') as mnist_pickle:
    MNIST = pickle.load(mnist_pickle, encoding='latin1')

# Распаковка данных
(train_features, train_labels), (val_features, val_labels), (test_features, test_labels) = MNIST

# Нормализация
train_features = train_features.astype(np.float32) / 256.0
val_features = val_features.astype(np.float32) / 256.0
test_features = test_features.astype(np.float32) / 256.0

# Функция для обучения
def train(positive_examples, negative_examples, num_iterations=100):
    num_dims = positive_examples.shape[1]
    weights = np.random.randn(num_dims + 1, 1) * 0.01  # Инициализация случайными значениями

    pos_count = positive_examples.shape[0]
    neg_count = negative_examples.shape[0]

    report_frequency = 10

    for i in range(num_iterations):
        pos = random.choice(positive_examples)
        neg = random.choice(negative_examples)

        z = np.dot(pos, weights[:-1])  # Используем только веса без смещения
        if z < 0:
            weights[:-1] += pos.reshape(weights[:-1].shape)

        z = np.dot(neg, weights[:-1])
        if z >= 0:
            weights[:-1] -= neg.reshape(weights[:-1].shape)

        if i % report_frequency == 0:
            pos_out = np.dot(positive_examples, weights[:-1])
            neg_out = np.dot(negative_examples, weights[:-1])
            pos_correct = (pos_out >= 0).sum() / float(pos_count)
            neg_correct = (neg_out < 0).sum() / float(neg_count)
            print("Итерация={}, правильных положительных={}, правильных отрицательных={}".format(i, pos_correct, neg_correct))

    return weights

# Функция для вычисления точности
def accuracy(weights, test_x, test_labels):
    # Добавление столбца единиц для учета смещения
    test_x_with_bias = np.c_[test_x, np.ones(len(test_x))]
    res = np.dot(test_x_with_bias, weights)
    return (res.reshape(test_labels.shape) * test_labels >= 0).sum() / float(len(test_labels))

# Задание положительных и отрицательных примеров для обучения
positive_examples = train_features[train_labels == 1]  # Пример для цифры 1
negative_examples = train_features[train_labels != 1]

# Обучение модели
weights = train(positive_examples, negative_examples)

# Вычисление точности
test_accuracy = accuracy(weights, test_features, test_labels)
print("Точность:", test_accuracy)

# Отображение примеров
fig = plt.figure(figsize=(10, 5))
for i in range(10):
    ax = fig.add_subplot(1, 10, i + 1)
    plt.imshow(train_features[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Метка: {train_labels[i]}')
    ax.axis('off')
plt.show()
