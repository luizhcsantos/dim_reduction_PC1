import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo


def load_data(fetch_function, target_column='target'):
    """
    Carrega e retorna o DataFrame usando uma função de fetch e especifica a coluna de destino.

    Parameters:
    - fetch_function: função que retorna um objeto com data.features e data.targets.
    - target_column: nome da coluna que conterá as classes/labels.

    Returns:
    - DataFrame com dados e uma única coluna de destino.
    """
    data = fetch_function()
    df = data.data.features.copy()

    # Verifica se `data.targets` é uma série ou DataFrame com múltiplas colunas
    if isinstance(data.data.targets, pd.DataFrame) and data.data.targets.shape[1] > 1:
        # Caso `targets` tenha múltiplas colunas, converte-as para uma única coluna, se necessário
        # Aqui, vamos apenas pegar a primeira coluna para `target`, mas você pode ajustar conforme necessário
        df[target_column] = data.data.targets.iloc[:, 0]
    else:
        df[target_column] = data.data.targets

    return df


def split_data(df, target_column='target', test_size=0.2, remove_prefix=None):
    """
    Divide o DataFrame em conjuntos de treino e teste, removendo o prefixo de classes, se necessário.

    Parameters:
    - df: DataFrame completo.
    - target_column: coluna com o rótulo das classes.
    - test_size: proporção de teste.
    - remove_prefix: prefixo para remover da coluna de classes (por exemplo, "Iris-").

    Returns:
    - x_train, y_train, x_test, y_test
    """
    train_set, test_set = train_test_split(df, test_size=test_size)
    x_train = train_set.drop(columns=[target_column])
    y_train = train_set[target_column].copy()
    x_test = test_set.drop(columns=[target_column])
    y_test = test_set[target_column].copy()

    # Remove o prefixo da coluna de destino, se necessário
    if remove_prefix:
        y_train = y_train.str.replace(f'^{remove_prefix}', '', regex=True)
        y_test = y_test.str.replace(f'^{remove_prefix}', '', regex=True)

    return x_train, y_train, x_test, y_test


def encode_classes(df, target_column='target', prefix='class'):
    """
    Aplica one-hot encoding à coluna de destino e outras colunas categóricas no DataFrame.

    Parameters:
    - df: DataFrame com a coluna de destino e outras colunas categóricas.
    - target_column: coluna que contém as classes/labels.
    - prefix: prefixo a ser adicionado nas colunas codificadas.

    Returns:
    - DataFrame com one-hot encoding aplicado a todas as colunas categóricas.
    """
    # Aplicar one-hot encoding na coluna de destino
    df_one_hot = pd.get_dummies(df[target_column], prefix=prefix)

    # Aplicar one-hot encoding nas outras colunas categóricas (não numéricas)
    categorical_columns = df.select_dtypes(include=['object']).columns
    df_one_hot_all = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    return df_one_hot_all


def plot_correlation_matrix(df, title):
    """
    Gera e plota a matriz de correlação para o DataFrame fornecido.

    Parameters:
    - df: DataFrame numérico.
    """
    correlation_matrix = df.corr()
    plt.figure(figsize=(20, 18))
    sns.set_theme(font_scale=1.2)
    sns.heatmap(correlation_matrix, cbar=True, annot=True, square=True, fmt='.2f',
                annot_kws={'size': 10}, linewidths=.5, cmap="coolwarm",
                yticklabels=df.columns, xticklabels=df.columns)
    plt.title("Matriz de Correlação", fontsize=16)
    plt.tight_layout()
    plt.savefig('mat_corr' + '_' + title+'.png')


def apply_pca_and_plot(x_train, y_train, dataset_cod, n_components=2, labels=None, colors=None):
    """
    Aplica PCA nos dados de treino e plota as duas primeiras componentes principais.

    Parameters:
    - x_train: conjunto de dados de treino.
    - y_train: rótulos de treino.
    - n_components: número de componentes principais para PCA.
    - labels: lista de rótulos das classes (para a legenda).
    - colors: lista de cores para as classes.
    """
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)

    df_pca = pd.DataFrame(data=x_train_pca, columns=[f'PC{i + 1}' for i in range(n_components)])

    unique_classes = y_train.unique()
    if labels is None:
        labels = unique_classes
    if colors is None:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Componente Principal 1', fontsize=10)
    ax.set_ylabel('Componente Principal 2', fontsize=10)
    ax.set_title('CP1 por CP2', fontsize=10)

    for target, color, label in zip(unique_classes, colors, labels):
        indices_to_keep = (y_train == target).values
        ax.scatter(df_pca.loc[indices_to_keep, 'PC1'],
                   df_pca.loc[indices_to_keep, 'PC2'],
                   c=[color], label=label, s=50)

    ax.legend(prop={'size': 5})
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    plt.savefig('pca_plot' + '_' + dataset_cod + '.png')


def main():
    # Execução do pipeline com parâmetros genéricos
    # id_repo = 53
    # prefix_to_remove = 'Iris-'
    # dataset_cod = 'Iris'
    # labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    id_repo = 320
    prefix_to_remove = ''
    dataset_cod = 'student_perf'

    df = load_data(fetch_function=lambda: fetch_ucirepo(id=id_repo), target_column='target')
    x_train, y_train, x_test, y_test = split_data(df, target_column='target', remove_prefix=prefix_to_remove)
    df_encoded = encode_classes(df, target_column='target')

    # non_numeric_columns = df_encoded.select_dtypes(exclude=[np.number]).columns
    # print("Colunas não numéricas:", non_numeric_columns)

    labels = [df_encoded.columns]

    plot_correlation_matrix(df_encoded, 'student')
    apply_pca_and_plot(x_train, y_train, dataset_cod , labels=labels)


if __name__ == "__main__":
    main()


