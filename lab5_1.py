import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz


# Загрузка данных из CSV файла
data = pd.read_csv('diabetes.csv')

# Выбор целевого атрибута и набора факторов
features = ['Glucose', 'Insulin', 'Age']
target = 'DiabetesPedigreeFunction'
X = data[features]
y = data[target]

# Создание экземпляра регрессора с разными критериями
models = {'gini': DecisionTreeRegressor(criterion='absolute_error'),
          'entropy': DecisionTreeRegressor(criterion='poisson'),
          'log_loss': DecisionTreeRegressor(criterion='squared_error')}

# Построение моделей и оценка их score
scores = {}
for criterion, model in models.items():
    model.fit(X, y)
    score = model.score(X, y)
    scores[criterion] = score

    # Визуализация получившегося дерева
    plt.figure(figsize=(10, 6))
    tree.plot_tree(model, feature_names=features, filled=True)
    plt.title(f'Decision Tree - Criterion: {criterion}')
    plt.show()

# Вывод оценок score моделей
for criterion, score in scores.items():
    print(f'Score for criterion {criterion}: {score}')























# # Выбор целевого атрибута и набора факторов
# target = data['DiabetesPedigreeFunction']
# features = data[['Glucose', 'Insulin', 'Age']]
#
# # Разделение данных на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
#
# # Список критериев
# criteria = ['gini', 'entropy', 'log_loss']
#
# # Список для сохранения оценок score моделей
# scores = []
#
# # Построение моделей и вычисление оценок score
# for criterion in criteria:
#     model = DecisionTreeClassifier(criterion=criterion)
#     # model = SVR()
#     # model.fit(X_train, y_train)
#     # Создание и обучение модели SVM
#     # model = SVR(kernel='linear')
#     model.
#     model.fit_transform(X_train, y_train)
#     svr_linear_score = model.score(X_test, y_test)
#     print("SVR (Linear Kernel) Score:", svr_linear_score)
#     y_pred = model.predict(X_test)
#     score = accuracy_score(y_test, y_pred)
#     scores.append(score)
#
#     # Визуализация получившегося дерева
#     plt.figure(figsize=(10, 8))
#     plot_tree(model, filled=True)
#     plt.title('Decision Tree with {} Criterion'.format(criterion))
#     plt.show()
#
# # Построение графика оценок score
# plt.bar(criteria, scores)
# plt.title('Score for Decision Tree with Different Criteria')
# plt.xlabel('Criterion')
# plt.ylabel('Score')
# plt.show()


# # Определение набора факторов (X) и целевого атрибута (y)
# X = data[['Glucose', 'Insulin', 'Age']]
# y = data['DiabetesPedigreeFunction']
#
# # Разделение данных на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Создание модели дерева решений с критерием Gini
# model = DecisionTreeClassifier(criterion='gini')
#
# model = LogisticRegression()
# # Обучение модели
# model.fit(X_train, y_train)
#
# # Предсказание классов для тестовых данных
# y_pred = model.predict(X_test)
#
# # Вычисление оценки score модели с критерием Gini
# score = accuracy_score(y_test, y_pred)
# print("Score модели (Gini):", score)
#
# # Визуализация дерева решений с критерием Gini
# dot_data = tree.export_graphviz(model, out_file=None, feature_names=X.columns, class_names=y.unique(), filled=True)
# graph = graphviz.Source(dot_data)
# graph.render("decision_tree_gini", format="png", cleanup=True)
# graph.view()
