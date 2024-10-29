import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def load_and_prepare_data():
    """Carrega o conjunto de dados Iris e prepara o DataFrame."""
    iris = fetch_ucirepo(id=53)
    X = iris.data.features
    y = iris.data.targets
    df = X.copy()
    df['target'] = y
    return df


def split_data(df):
    """Divide os dados em conjuntos de treino e teste."""
    train_set, test_set = train_test_split(df, test_size=0.2)
    x_train = train_set.drop(columns=['target'])
    y_train = train_set['target'].copy()
    x_test = test_set.drop(columns=['target'])
    y_test = test_set['target'].copy()

    # Removendo o prefixo 'Iris-' das classes
    y_train = y_train.str.replace(r'^Iris-', '', regex=True)
    y_test = y_test.str.replace(r'^Iris-', '', regex=True)

    return x_train, y_train, x_test, y_test


def encode_classes(df):
    """Aplica one-hot encoding às classes no DataFrame."""
    df_one_hot = pd.get_dummies(df['target'], prefix='class')
    df_encoded = pd.concat([df, df_one_hot], axis=1)
    return df_encoded.drop(columns=['target'])


def plot_correlation_matrix(df):
    """Gera e plota a matriz de correlação para o DataFrame fornecido."""
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    sns.set_theme(font_scale=1.2)
    sns.heatmap(correlation_matrix, cbar=True, annot=True, square=True, fmt='.2f',
                annot_kws={'size': 10}, linewidths=.5, cmap="coolwarm",
                yticklabels=df.columns, xticklabels=df.columns)
    plt.title("Matriz de Correlação", fontsize=16)
    plt.tight_layout()
    plt.savefig('mat_corr.png')


def apply_pca_and_plot(x_train, y_train):
    """Aplica PCA nos dados de treino e plota as duas primeiras componentes principais."""
    pca = PCA(n_components=4)
    x_train_pca = pca.fit_transform(x_train)

    df_pca = pd.DataFrame(data=x_train_pca, columns=[f'PC{i + 1}' for i in range(4)])

    targets = ['setosa', 'versicolor', 'virginica']
    labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    colors = ['r', 'g', 'b']

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Componente Principal 1', fontsize=10)
    ax.set_ylabel('Componente Principal 2', fontsize=10)
    ax.set_title('CP1 por CP2', fontsize=10)

    for target, color, label in zip(targets, colors, labels):
        indicesToKeep = (y_train == target).values
        ax.scatter(df_pca.loc[indicesToKeep, 'PC1'],
                   df_pca.loc[indicesToKeep, 'PC2'],
                   c=color, label=label, s=50)

    ax.legend(labels, prop={'size': 5})
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    plt.savefig('pca_iris1.png')

def main():
    # Execução do pipeline
    df = load_and_prepare_data()
    x_train, y_train, x_test, y_test = split_data(df)
    df_encoded = encode_classes(df)
    plot_correlation_matrix(df_encoded)
    apply_pca_and_plot(x_train, y_train)

if __name__ == "__main__":
    main()