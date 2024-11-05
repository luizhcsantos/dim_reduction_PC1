import matplotlib.pyplot as plt
import numpy as np
import pacmap
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
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
        # Aqui, vamos apenas pegar a primeira coluna para `target`
        df[target_column] = data.data.targets.iloc[:, 0]
    else:
        df[target_column] = data.data.targets

    return df


def split_data(df, target_column='target', test_size=0.2):
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


def plot_correlation_matrix(df, title):
    """
    Gera e plota a matriz de correlação para o DataFrame fornecido.

    Parameters:
    - df: DataFrame numérico.
    """
    correlation_matrix = df.corr()


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



def main():
    # Execução do pipeline com parâmetros genéricos

    dataset_cod = 'student_perf'

    # data = load_data(fetch_function=lambda: fetch_ucirepo(id=320), target_column='target')
    # data.to_csv('ucirepo_student.csv', index=False)

    # print(data.info())
    data = pd.read_csv('ucirepo_student.csv')

    data['sex'] = data['sex'].map({'male': 0, 'female': 1})
    data['Pstatus'] = data['Pstatus'].map({'together': 0, 'apart': 1})
    data['schoolsup'] = data['schoolsup'].map({'yes': 1, 'no': 0})
    data['famsup'] = data['famsup'].map({'yes': 1, 'no': 0})
    data['paid'] = data['paid'].map({'yes': 1, 'no': 0})
    data['activities'] = data['activities'].map({'yes': 1, 'no': 0})
    data['internet'] = data['internet'].map({'yes': 1, 'no': 0})
    data['nursery'] = data['nursery'].map({'yes': 1, 'no': 0})
    data['higher'] = data['higher'].map({'yes': 1, 'no': 0})
    data['romantic'] = data['romantic'].map({'yes': 1, 'no': 0})

    # One-Hot Encoding para variáveis com mais de duas categorias
    data = pd.get_dummies(data, columns=['school', 'guardian', 'reason', 'Mjob', 'Fjob'])

    # Selecione as features numéricas e as recém transformadas
    # Definimos a lista de features numéricas primeiro, depois incluímos as novas colunas do One-Hot Encoding
    numerical_features = [
        'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'freetime', 'goout',
        'Walc', 'Dalc', 'health', 'absences'
    ]

    # Verifique se G1, G2, G3 estão no DataFrame e adicione se estiverem presentes
    for grade in ['G1', 'G2', 'G3']:
        if grade in data.columns:
            numerical_features.append(grade)

    # Agora combinamos as features numéricas com todas as colunas do DataFrame atual
    features = numerical_features + [col for col in data.columns if col not in numerical_features]

    # Filtrar somente colunas numéricas em X
    x = data[features].select_dtypes(include=['float64', 'int64'])

    # Imputação dos valores faltantes com a média de cada coluna
    imputer = SimpleImputer(strategy='mean')
    x_imputed = imputer.fit_transform(x)

    # Padronização das features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)

    # Aplicar PCA
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)

    # Aplicar t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    x_tsne = tsne.fit_transform(x_scaled)

    # Aplicar UMAP
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    x_umap = umap_reducer.fit_transform(x_scaled)

    # Aplicar PaCMAP
    pacmap_reducer = pacmap.PaCMAP(n_components=2, random_state=42)
    x_pacmap = pacmap_reducer.fit_transform(x_scaled)

    # Visualização dos resultados
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Definir o color map
    color_map = data['G3'] if 'G3' in data.columns else x_pca[:, 1]

    # Plot e salvamento de cada gráfico
    # PCA plot
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=color_map, cmap='viridis')
    plt.colorbar(sc)
    plt.title("PCA")
    plt.savefig("PCA_plot.png")
    plt.close()  # Fecha a figura para liberar memória

    # t-SNE plot
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=color_map, cmap='viridis')
    plt.colorbar(sc)
    plt.title("t-SNE")
    plt.savefig("tSNE_plot.png")
    plt.close()

    # UMAP plot
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x_umap[:, 0], x_umap[:, 1], c=color_map, cmap='viridis')
    plt.colorbar(sc)
    plt.title("UMAP")
    plt.savefig("UMAP_plot.png")
    plt.close()

    # PaCMAP plot
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x_pacmap[:, 0], x_pacmap[:, 1], c=color_map, cmap='viridis')
    plt.colorbar(sc)
    plt.title("PaCMAP")
    plt.savefig("PaCMAP_plot.png")
    plt.close()

if __name__ == "__main__":
    main()


