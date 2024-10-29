import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn import datasets


def main():
    # dataset = pd.read_csv('wine.csv')

    
    dataset = datasets.load_iris()

    # distributing the dataset into two components X and Y
    X = dataset.iloc.values
    y = dataset.iloc.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    pca = PCA(n_components=2)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    explained_variance = pca.explained_variance_ratio_

    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    training_results(X_train, y_train, classifier)

    test_results(X_test, y_test, classifier)

    pca_space(X_train, y_train, y)


def training_results(X_train, y_train, classifier): 

    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                                stop=X_set[:, 0].max() + 1, step=0.01),
                        np.arange(start=X_set[:, 1].min() - 1,
                                stop=X_set[:, 1].max() + 1, step=0.01))

    plt.clf()
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
                                                    X2.ravel()]).T).reshape(X1.shape), alpha=0.75,
                cmap=ListedColormap(('yellow', 'white', 'aquamarine')))

    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=ListedColormap(('red', 'green', 'blue'))(i), label=j)

    plt.title('Logistic Regression (Training set)')
    plt.xlabel('PC1')  # for Xlabel
    plt.ylabel('PC2')  # for Ylabel
    plt.legend()  # to show legend

    plt.savefig('train_set.png')

    # # show scatter plot
    # plt.show()

def test_results(X_test, y_test, classifier): 
    # Visualising the Test set results through scatter plot
    X_set, y_set = X_test, y_test

    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                                stop=X_set[:, 0].max() + 1, step=0.01),
                        np.arange(start=X_set[:, 1].min() - 1,
                                stop=X_set[:, 1].max() + 1, step=0.01))
    
    plt.clf()
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
                                                    X2.ravel()]).T).reshape(X1.shape), alpha=0.75,
                cmap=ListedColormap(('yellow', 'white', 'aquamarine')))

    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=ListedColormap(('red', 'green', 'blue'))(i), label=j)

    # title for scatter plot
    plt.title('Logistic Regression (Test set)')
    plt.xlabel('PC1')  # for Xlabel
    plt.ylabel('PC2')  # for Ylabel
    plt.legend()

    plt.savefig('test_set.png')

    # # show scatter plot
    # plt.show()

def pca_space(X_train, y_train, y): 

    colors = ["r", "g", "b"]
    labels = ["Class 1", "Class 2", "Class 3"]
    plt.clf()
    for i, color, label in zip(np.unique(y), colors, labels):
        plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], color=color, label=label)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.savefig('pca_space.png')
    # plt.show()

if __name__ == "__main__":
    main()