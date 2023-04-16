import numpy as np

damping_factor = 0.8
# Просим пользователя ввести количество вершин в графе
print("Введите количество элементов n")
size = int(input())
adj_matrix = np.array([[0.0] * size for i in range(size)])

# Составляем матрицу переходов для заданного количества вершин
for i in range(size):
    for j in range(size):
        if i == 0 and j == 0:
            adj_matrix[0][0] = 0.5
        elif i == 1 and j == 0:
            adj_matrix[i][j] = 0.5
        elif j == i - 1:
            adj_matrix[i][j] = 1
        else:
            adj_matrix[i][j] = 0

# for i in range(size):
#     for j in range(size):
#         print(adj_matrix[i][j], end=" ")
#     print()

# Создаем вектор начальных вероятностей
v = np.array([1 / size] * size)
# Создаем единичный вектор e
e = np.array([1] * size)

# Итеративно вычисляем PageRank
for i in range(100):
    r = damping_factor * np.dot(adj_matrix, v) + ((1 - damping_factor) / size) * e
    v = r
# Выводим получившийся вектор v, содержащий pagerank для каждой вершины
for i in range(v.size):
    print(v[i], end=" ")

