import matplotlib.pyplot as plt
import numpy as np
import pacmap
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
import umap.umap_ as umap
from sklearn.manifold import Isomap
from ucimlrepo import fetch_ucirepo
from scipy.spatial import distance_matrix
from sklearn.metrics import silhouette_score


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

    # data = pd.read_csv('ucirepo_student.csv')

    data = fetch_ucirepo(id=320)
    x = data.data.features
    y = data.data.targets

    #
    # # metadata
    # print(student_performance.metadata)
    #
    # # variable information

    #  .loc[row_indexer,col_indexer] = value

    x.loc[:, 'sex'] = x['sex'].map({'male': 0, 'female': 1})
    x.loc[:, 'address'] = x['address'].map({'U': 0, 'R': 1})
    x.loc[:,'famsize'] = x['famsize'].map({'LE3': 0, 'GT3': 1})
    x.loc[:, 'Pstatus'] = x['Pstatus'].map({'together': 0, 'apart': 1})
    x.loc[:, 'schoolsup'] = x['schoolsup'].map({'yes': 1, 'no': 0})
    x.loc[:, 'famsup'] = x['famsup'].map({'yes': 1, 'no': 0})
    x.loc[:, 'paid'] = x['paid'].map({'yes': 1, 'no': 0})
    x.loc[:, 'activities'] = x['activities'].map({'yes': 1, 'no': 0})
    x.loc[:, 'internet'] = x['internet'].map({'yes': 1, 'no': 0})
    x.loc[:, 'nursery'] = x['nursery'].map({'yes': 1, 'no': 0})
    x.loc[:, 'higher'] = x['higher'].map({'yes': 1, 'no': 0})
    x.loc[:, 'romantic'] = x['romantic'].map({'yes': 1, 'no': 0})
    #
    # # One-Hot Encoding para variáveis com mais de duas categorias
    x = pd.get_dummies(x, columns=['school', 'guardian', 'reason', 'Mjob', 'Fjob'], drop_first=False)
    #
    # # Selecione as features numéricas e as recém transformadas
    # # Definimos a lista de features numéricas primeiro, depois incluímos as novas colunas do One-Hot Encoding
    # numerical_features = [
    #     'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'freetime', 'goout',
    #     'Walc', 'Dalc', 'health', 'absences'
    # ]
    #
    # # Verifique se G1, G2, G3 estão no DataFrame e adicione se estiverem presentes
    # for grade in ['G1', 'G2', 'G3']:
    #     if grade in data.columns:
    #         numerical_features.append(grade)
    #
    #
    # # Agora combinamos as features numéricas com todas as colunas do DataFrame atual
    # features = numerical_features + [col for col in data.columns if col not in numerical_features]
    #
    # # Filtrar somente colunas numéricas em X
    # x = data[features].select_dtypes(include=['float64', 'int64'])

    # Imputação dos valores faltantes com a média de cada coluna
    imputer = SimpleImputer(strategy='mean')
    x_imputed = imputer.fit_transform(x)
    #
    # # Padronização das features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)

    # # Visualização dos resultados
    # fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    #
    # plt.figure(figsize=(8, 6))

    # Aplicar PCA
    # pca = PCA(n_components=15, random_state=42)
    # x_pca = pca.fit_transform(x_scaled)
    #
    # # Extrair a variância explicada por cada componente principal
    # explained_variance_ratio = pca.explained_variance_ratio_
    #
    # # Aproximar a variância explicada pelas duas primeiras componentes
    # variance_2_components = explained_variance_ratio[0] + explained_variance_ratio[1]
    # print(f"Variância explicada pelas duas primeiras componentes: {variance_2_components * 100:.2f}%")
    #
    # # Suponha que o PCA já tenha sido aplicado e você tenha o explained_variance_ratio_
    # cumulative_variance = np.cumsum(explained_variance_ratio)
    # #
    # # # Plot da variância acumulada
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='b')
    # plt.xlabel('Número de Componentes Principais')
    # plt.ylabel('Variância Acumulada')
    # plt.title('Variância Acumulada por Número de Componentes Principais')
    # plt.axhline(y=0.9, color='r', linestyle='-', label='80% da Variância Explicada')  # Linha de referência para 90% da variância explicada
    # plt.axhline(y=0.8, color='g', linestyle='-', label='90% da Variância Explicada')  # Linha de referência para 80% da variância explicada
    # plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
    # plt.grid(True)
    # # plt.savefig("PCA_variancia_PCs_1.png")
    # plt.show()
    #
    # # Encontrar o número de componentes para 80% e 90% de variância explicada
    # num_components_80 = np.argmax(cumulative_variance >= 0.8) + 1
    # num_components_90 = np.argmax(cumulative_variance >= 0.9) + 1
    #
    # print(f"Número de componentes para explicar 80% da variância: {num_components_80}")
    # print(f"Número de componentes para explicar 90% da variância: {num_components_90}")

    # # Plotar um histograma das variâncias explicadas
    # plt.figure(figsize=(8, 5))
    # plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, color='b')
    # plt.xlabel('Componentes Principais')
    # plt.ylabel('Variância Explicada')
    # plt.title('Variância Explicada por Cada Componente Principal')
    # plt.xticks(range(1, len(explained_variance_ratio) + 1))
    # plt.savefig("PCA_variancia_PCs.png")
    # plt.show()


    # # Criar o gráfico de dispersão usando a coluna 'G3' como mapa de cores
    # plt.figure(figsize=(8, 6))
    # color_map = y['G3']
    # sc = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=color_map, cmap='plasma', alpha=0.7)
    # plt.colorbar(sc, label="G3")
    # plt.xlabel("PCA Component 1")
    # plt.ylabel("PCA Component 2")
    # plt.title("PCA with G3 as Color Map")
    # # plt.savefig('pca_plot_new_15pcs.png')
    # plt.show()


    # x = data.drop(columns=['target'])  # Remova a coluna 'target' para aplicar o t-SNE nas features
    # y = data['target']  # Defina o 'target' como y


    # Aplicar t-SNE
    # valor padrão para perplexidade = 30
    # tsne = TSNE(perplexity=30, random_state=42)
    # x_tsne = tsne.fit_transform(x_scaled)

    # df = pd.DataFrame({
    #     't-SNE1': x_tsne[:, 0],
    #     't-SNE2': x_tsne[:, 1],
    #     'PCA1': x_pca[:, 0],
    #     'PCA2': x_pca[:, 1],
    #     'UMAP1': x_umap[:, 0],
    #     'UMAP2': x_umap[:, 1],
    #     'Class': y
    # })
    #
    # sns.pairplot(df, hue='Class', vars=['t-SNE1', 't-SNE2', 'PCA1', 'PCA2', 'UMAP1', 'UMAP2'])
    # plt.suptitle("Pairplot of t-SNE, PCA, and UMAP", y=1.02)
    # plt.savefig('tsne_pairplot.png')
    # plt.show()

    # Definir o color map
    color_map = y['G3']

    # t-SNE plot
    # plt.figure(figsize=(6, 5))
    # sc = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=color_map, cmap='plasma')
    # plt.colorbar(sc)
    # plt.title("t-SNE")
    # plt.savefig("tsne_plot_30new.png")
    # plt.close()

    # Configurar a figura com subplots lado a lado
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    #
    # # Aplicar t-SNE com perplexidade de 50
    # tsne_50 = TSNE(n_components=2, perplexity=50, random_state=42)
    # x_tsne_50 = tsne_50.fit_transform(x_scaled)
    #
    # # Primeiro subplot (perplexity=50)
    # sc_50 = axs[0].scatter(x_tsne_50[:, 0], x_tsne_50[:, 1], c=color_map, cmap='plasma', alpha=0.7)
    # axs[0].set_title("t-SNE com Perplexidade 50")
    # axs[0].set_xlabel("Dimensão 1")
    # axs[0].set_ylabel("Dimensão 2")
    # fig.colorbar(sc_50, ax=axs[0])
    #
    # # Aplicar t-SNE com perplexidade de 20
    # tsne_20 = TSNE(n_components=2, perplexity=20, random_state=42)
    # x_tsne_20 = tsne_20.fit_transform(x_scaled)
    #
    # # Segundo subplot (perplexity=20)
    # sc_20 = axs[1].scatter(x_tsne_20[:, 0], x_tsne_20[:, 1], c=color_map, cmap='plasma', alpha=0.7)
    # axs[1].set_title("t-SNE com Perplexidade 20")
    # axs[1].set_xlabel("Dimensão 1")
    # axs[1].set_ylabel("Dimensão 2")
    # fig.colorbar(sc_20, ax=axs[1])
    #
    # # Ajustar layout e salvar a figura
    # plt.tight_layout()
    # plt.savefig("tSNE_plot_perplexity_comparison.png")
    # plt.show()

    # Suponha que x_tsne seja o resultado do t-SNE em duas dimensões
    # sns.kdeplot(x=x_tsne[:, 0], y=x_tsne[:, 1], cmap="magma", fill=True, bw_adjust=0.7)
    # plt.xlabel("t-SNE Dimension 1")
    # plt.ylabel("t-SNE Dimension 2")
    # plt.title("t-SNE Density Plot")
    # plt.savefig('tSNE_plot_densidade.png')
    # plt.show()

    # Calcula a matriz de distância e define um limite de vizinhança
    # dist_matrix = distance_matrix(x_tsne, x_tsne)
    # threshold = np.percentile(dist_matrix,5)  # Exemplo: manter apenas as conexões mais próximas (5% menores distâncias)
    #
    # plt.figure(figsize=(8, 6))
    # for i in range(len(x_tsne)):
    #     for j in range(i + 1, len(x_tsne)):
    #         if dist_matrix[i, j] < threshold:
    #             plt.plot([x_tsne[i, 0], x_tsne[j, 0]], [x_tsne[i, 1], x_tsne[j, 1]], 'k-', lw=0.2, alpha=0.5)
    #
    # plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c='b', s=10)
    # plt.xlabel("t-SNE Dimension 1")
    # plt.ylabel("t-SNE Dimension 2")
    # plt.title("t-SNE with Nearest Neighbor Connections")
    # plt.savefig('tSNE_conexao_vizinhos_proximos.png')
    # plt.show()

    # Aplicar UMAP
    # umap_reducer = umap.UMAP(random_state=42)
    # x_umap = umap_reducer.fit_transform(x_scaled)
    # #
    # UMAP plot
    # color_map = y['G3']
    # plt.figure(figsize=(6, 5))
    # sc = plt.scatter(x_umap[:, 0], x_umap[:, 1], c=color_map, cmap='plasma')
    # plt.colorbar(sc)
    # plt.title("UMAP")
    # plt.savefig("umap_plot_new.png")
    # plt.close()

    # fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Aplicar UMAP com numero de vizinhos padrão 15
    # umap_15 = umap.UMAP(random_state=42)
    # x_umap_15 = umap_15.fit_transform(x_scaled)
    #
    # color_map = y['G3']
    #
    #
    # # # Primeiro subplot (n_neighbors=15)
    # # color_map_15 = data['G3'] if 'G3' in data.columns else x_umap_15[:, 1]
    # sc_15 = axs[0].scatter(x_umap_15[:, 0], x_umap_15[:, 1], c=color_map, cmap='plasma', alpha=0.7)
    # axs[0].set_title("UMAP com parâmetro n_neighbors padrão 15")
    # axs[0].set_xlabel("Dimensão 1")
    # axs[0].set_ylabel("Dimensão 2")
    # fig.colorbar(sc_15, ax=axs[0])
    #
    # # Aplicar UMAP com numero de vizinhos 50
    # umap_50 = umap.UMAP(n_neighbors=50, random_state=42)
    # x_umap_50 = umap_50.fit_transform(x_scaled)
    #
    #
    # # Segundo subplot (n_neighbors=50)
    # # color_map_50 = data['G3'] if 'G3' in data.columns else x_umap_50[:, 1]
    # sc_50 = axs[1].scatter(x_umap_50[:, 0], x_umap_50[:, 1], c=color_map, cmap='plasma', alpha=0.7)
    # axs[1].set_title("UMAP com parâmetro n_neighbors = 50")
    # axs[1].set_xlabel("Dimensão 1")
    # axs[1].set_ylabel("Dimensão 2")
    # fig.colorbar(sc_50, ax=axs[1])
    #
    # # Aplicar UMAP com numero de vizinhos 10
    # umap_100 = umap.UMAP(n_neighbors=100, random_state=42)
    # x_umap_100 = umap_100.fit_transform(x_scaled)
    #
    #
    # # Terceiro subplot (n_neighbors=100)
    # # color_map_100 = data['G3'] if 'G3' in data.columns else x_umap_100[:, 1]
    # sc_100 = axs[2].scatter(x_umap_100[:, 0], x_umap_100[:, 1], c=color_map, cmap='plasma', alpha=0.7)
    # axs[2].set_title("UMAP com parâmetro n_neighbors = 100")
    # axs[2].set_xlabel("Dimensão 1")
    # axs[2].set_ylabel("Dimensão 2")
    # fig.colorbar(sc_100, ax=axs[2])

    # Ajustar layout e salvar a figura
    # plt.tight_layout()
    # plt.savefig("umap_neighbors_comp_new.png")
    # plt.show()

    # Aplicar PaCMAP
    # pacmap_reducer = pacmap.PaCMAP(n_components=2, random_state=42)
    # x_pacmap = pacmap_reducer.fit_transform(x_scaled)
    #
    # # PaCMAP plot
    # color_map = data['G3'] if 'G3' in data.columns else x_pacmap[:, 1]
    # plt.figure(figsize=(6, 5))
    # sc = plt.scatter(x_pacmap[:, 0], x_pacmap[:, 1], c=color_map, cmap='Paired')
    # plt.colorbar(sc)
    # plt.title("PaCMAP")
    # plt.savefig("PaCMAP_plot.png")
    # plt.close()

    # Aplicar Isomap
    # isomap = Isomap(n_components=2, n_neighbors=15)
    # x_isomap = isomap.fit_transform(x_scaled)
    #
    # # Plotar o resultado do Isomap
    # color_map = data['G3'] if 'G3' in data.columns else x_isomap[:, 1]
    # plt.figure(figsize=(8, 6))
    # sc = plt.scatter(x_isomap[:, 0], x_isomap[:, 1], c=color_map, cmap='Paired', alpha=0.7)
    # plt.colorbar(sc)
    # plt.xlabel("Isomap Dimension 1")
    # plt.ylabel("Isomap Dimension 2")
    # plt.title("Isomap Result")
    # plt.savefig('isomap_plot4.png')
    # plt.show()

    # Aplicar Kernel PCA
    # kernelpca = KernelPCA(n_components=7, kernel='linear', random_state=42)
    # x_kpca = kernelpca.fit_transform(x_scaled)
    #
    # # Plotar o resultado do Kernel PCA
    # color_map = data['G3'] if 'G3' in data.columns else x_kpca[:, 1]
    # plt.figure(figsize=(8, 6))
    # sc = plt.scatter(x_kpca[:, 0], x_kpca[:, 1], c=color_map, cmap='Paired', alpha=0.7)
    # plt.colorbar(sc)
    # plt.xlabel("KernelPCA Dimension 1")
    # plt.ylabel("KernelPCA Dimension 2")
    # plt.title("Kernel PCA")
    # plt.savefig('kpca_plot1.png')
    # plt.show()

    # Aplicar LLE
    # lle = LocallyLinearEmbedding(n_components=2, n_neighbors=15,  random_state=42)
    # x_lle = lle.fit_transform(x_scaled)
    #
    # # # Plotar o resultado do LLE
    # color_map = data['G3'] if 'G3' in data.columns else x_lle[:, 1]
    # plt.figure(figsize=(8, 6))
    # sc = plt.scatter(x_lle[:, 0], x_lle[:, 1], c=color_map, cmap='Paired', alpha=0.7)
    # plt.colorbar(sc)
    # plt.xlabel("LLE Dimension 1")
    # plt.ylabel("LLE Dimension 2")
    # plt.title("Locally Linear Embedding")
    # plt.savefig('lle_plot1.png')
    # plt.show()


    # Aplicar PCA
    # pca = PCA(n_components=2)
    # x_pca = pca.fit_transform(x_scaled)
    #
    # # Aplicar LLE
    # lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, method='standard', random_state=42)
    # x_lle = lle.fit_transform(x_scaled)
    #
    # # Aplicar Isomap
    # isomap = Isomap(n_neighbors=10, n_components=2)
    # x_isomap = isomap.fit_transform(x_scaled)
    #
    # # Plotar os resultados
    # fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    #
    # # Plot PCA
    # sc1 = axs[0].scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='Paired', alpha=0.7)
    # axs[0].set_title("PCA")
    # axs[0].set_xlabel("Component 1")
    # axs[0].set_ylabel("Component 2")
    # fig.colorbar(sc1, ax=axs[0])
    #
    # # Plot LLE
    # sc2 = axs[1].scatter(x_lle[:, 0], x_lle[:, 1], c=y, cmap='Paired', alpha=0.7)
    # axs[1].set_title("LLE")
    # axs[1].set_xlabel("LLE Dimension 1")
    # axs[1].set_ylabel("LLE Dimension 2")
    # fig.colorbar(sc2, ax=axs[1])
    #
    # # Plot Isomap
    # sc3 = axs[2].scatter(x_isomap[:, 0], x_isomap[:, 1], c=y, cmap='Paired', alpha=0.7)
    # axs[2].set_title("Isomap")
    # axs[2].set_xlabel("Isomap Dimension 1")
    # axs[2].set_ylabel("Isomap Dimension 2")
    # fig.colorbar(sc3, ax=axs[2])
    #
    # # Ajustar layout e adicionar barra de cores
    # plt.tight_layout()
    # plt.show()
    #
    # # Cálculo de métricas quantitativas (exemplo com Coeficiente de Silhueta)
    # if y is not None:
    #     pca_silhouette = silhouette_score(x_pca, y)
    #     lle_silhouette = silhouette_score(x_lle, y)
    #     isomap_silhouette = silhouette_score(x_isomap, y)
    #     print(f"Silhouette Score for PCA: {pca_silhouette:.2f}")
    #     print(f"Silhouette Score for LLE: {lle_silhouette:.2f}")
    #     print(f"Silhouette Score for Isomap: {isomap_silhouette:.2f}")


    # Aplicar MDS
    mds_reducer = MDS(random_state=42)
    x_mds = mds_reducer.fit_transform(x_scaled)
    #
    # MDS plot
    color_map = y['G3']
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x_mds[:, 0], x_mds[:, 1], c=color_map, cmap='plasma')
    plt.colorbar(sc)
    plt.title("MDS")
    plt.savefig("mds_plot.png")
    plt.close()

if __name__ == "__main__":
    main()


