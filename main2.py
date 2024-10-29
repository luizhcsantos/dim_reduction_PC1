import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
df = iris.data.features.copy()
df['target'] = iris.data.targets

train_set, test_set = train_test_split(df, test_size=0.2) #, random_state=0)

x_train = train_set.drop(columns=['target'])
y_train = train_set['target'].copy()

x_test = test_set.drop(columns=['target'])
y_test = test_set['target'].copy()

y_train_copy = y_train.str.replace(r'^Iris-', '', regex=True)
y_test_copy = y_test.str.replace(r'Iris-', '', regex=True)

y_class_count = dict(y_train_copy.value_counts())
y_samples_sum = sum(y_class_count.values())


# print("Ocorrência de Iris-virginica: {0}".format(y_class_count['virginica']/y_samples_sum))
# print("Ocorrência de Iris-setosa: {0}".format(y_class_count['setosa']/y_samples_sum))
# print("Ocorrência de Iris-versicolor: {0}".format(y_class_count['versicolor']/y_samples_sum))

# Transformar as classes em colunas binárias
y_train_encoded = pd.get_dummies(y_train_copy, prefix='class')
y_test_encoded = pd.get_dummies(y_test_copy, prefix='class')

df_one_hot = pd.get_dummies(df['target'], prefix='class')
df = pd.concat([df, df_one_hot], axis=1)

df_numeric = df.drop(columns=['target'])

#gera uma matriz de correlação entre as variáveis independentes
correlation_matrix = df_numeric.corr()

#plota um heatmap da matriz, para facilitar a visualização
# plt.figure(figsize=(12, 10))  # Ajuste do tamanho da figura para se adequar melhor
# sns.set_theme(font_scale=1.2)  # Ajuste do tamanho da fonte
# hm = sns.heatmap(correlation_matrix,
#                  cbar=True,
#                  annot=True,
#                  square=True,
#                  fmt='.2f',  # Definindo o formato dos valores
#                  annot_kws={'size': 10},  # Ajustando o tamanho do texto de anotações
#                  linewidths=.5,  # Reduzindo a espessura das linhas entre blocos
#                  cmap="coolwarm",  # Mudando para uma paleta de cores mais clara
#                  yticklabels=df_numeric.columns,
#                  xticklabels=df_numeric.columns)
#
# plt.title("Matriz de Correlação", fontsize=16)
# plt.tight_layout()
# plt.savefig('mat_corr.png')

pca = PCA(n_components=4)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

df_pca = pd.DataFrame(data=x_train_pca)
print(df_pca.head())


targets = ['setosa', 'versicolor', 'virginica']
labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
colors = ['r', 'g', 'b']

fig = plt.figure(figsize = (6,6))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Componente Principal 1', fontsize = 10)
ax.set_ylabel('Componente Principal 2', fontsize = 10)
ax.set_title('CP1 por CP2', fontsize = 10)

for target, color, label in zip(targets, colors, labels):
    indicesToKeep = (y_train_copy == target).values
    ax.scatter((df_pca.loc[indicesToKeep, 0])
               ,(df_pca.loc[indicesToKeep, 1]),
               c = color,
               label = label,
               s=50)

ax.legend(labels, prop={'size': 5})
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
plt.savefig('pca_iris1.png')