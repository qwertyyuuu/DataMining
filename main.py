import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR


data = pd.read_csv('dataset.csv')

# выбор зависимой переменной и набора независимых переменных
dependent_var = 'pts'
independent_vars = ['gp', 'reb', 'ast']

# создание матрицы объекты-признаки
X = data[independent_vars].values
y = data[dependent_var].values

# разделение выборки на тренировочную и тестовую
train_size = 0.7
train_len = int(len(X) * train_size)
X_train, y_train = X[:train_len], y[:train_len]
X_test, y_test = X[train_len:], y[train_len:]

# создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# оценка score модели на тренировочной и тестовой выборках
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

# вывод результатов
print('Train score:', train_score)
print('Test score:', test_score)

# визуализация результатов
y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

# Создание полиномиальных признаков
degree = 2  # степень полинома
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

# Разделение выборки на тренировочную и тестовую
train_size2 = 0.7
train_len2 = int(len(X_poly) * train_size)
X_train2, y_train2 = X_poly[:train_len2], y[:train_len2]
X_test2, y_test2 = X_poly[train_len2:], y[train_len2:]

# Создание и обучение модели полиномиальной регрессии
model = LinearRegression()
model.fit(X_train2, y_train2)

# Оценка score модели на тренировочной и тестовой выборках
train_score2 = model.score(X_train2, y_train2)
test_score2 = model.score(X_test2, y_test2)

# Вывод результатов
print('Train score:', train_score2)
print('Test score:', test_score2)

# Построение визуализации
y_pred2 = model.predict(X_test2)

# plt.scatter(X_test2[:, 1], y_test2, color='b', label='Actual')
plt.scatter(X_test2[:, 1], y_pred2, color='b', label='Predicted')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()

X_train3, X_test3, y_train3, y_test3 = train_test_split(data[independent_vars], data[dependent_var], test_size=0.2)

# Создание и обучение моделей SVM с разными ядерными функциями
kernels = ['linear', 'poly', 'rbf']
scores = []
c = 0.05

for kernel in kernels:
    model = SVR(kernel=kernel)
    model.fit(X_train3, y_train3)
    y_pred = model.predict(X_test3)
    score = mean_squared_error(y_test3, y_pred) * c
    scores.append(score)
    print("Score модели (",kernel,"):", score)

    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('SVM with {} Kernel'.format(kernel))
    plt.xlabel('Samples')
    plt.ylabel('Target')
    plt.legend()
    plt.show()

# Визуализация оценок score моделей

plt.bar(kernels, scores)
plt.xlabel('Ядерная функция')
plt.ylabel('Score')
plt.title('Оценка score модели для разных ядерных функций')
plt.show()


# # Список ядерных функций для SVM
# kernel_functions = ['linear', 'poly', 'rbf', 'sigmoid']
#
# # Применение SVM с различными ядерными функциями
# for kernel in kernel_functions:
#     # Создание модели SVM
#     model3 = svm.SVR(kernel=kernel)
#
# # Обучение модели на тренировочном наборе
# model3.fit(X_train3, y_train3)
#
# # Прогнозирование целевого атрибута на тестовом наборе
# y_pred3 = model3.predict(X_test3)
#
# # # Оценка точности модели
# # accuracy = accuracy_score(y_test3, y_pred3)
# #
# # # Вывод оценки точности для каждого метода
# # print(f"Kernel: {kernel}, Accuracy: {accuracy}")
#
# # Визуализация результатов
#
# plt.scatter(X_test3[:, 1])
# plt.xlabel('pts')
# plt.title(f"SVM with {kernel} kernel")
# plt.show()