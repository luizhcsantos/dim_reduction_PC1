import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('df2.xlsx')

features = ['Mouse 1', 'Mouse 2', 'Mouse 3', 'Mouse 4', 'Mouse 5', 'Mouse 6']

x = df.loc[:, features].values

# Separating out the target
y = df.loc[:,['target']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=3)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data=principalComponents
                           , columns=['principal component 1', 'principal component 2', 'principal component 3'])

finalDf = pd.concat([principalDf, df[['target']]], axis=1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['Gene 1', 'Gene 2', 'Gene 3', 'Gene 4']
colors = ['yellow', 'green', 'black', 'purple']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

plt.show()
fig.savefig('pcaPC1-2.png')

# Bar plot of explained_variance
# plt.bar(
#     range(1,len(pca.explained_variance_)+1),
#     pca.explained_variance_
#     )
#
# plt.xlabel('PCA Feature')
# plt.ylabel('Explained variance')
# plt.title('Feature Explained Variance')
# plt.show()