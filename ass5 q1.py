from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")
df= pd.read_csv(r"C:\Users\srava\PycharmProjects\assignment 5\datasets\CC.csv")
print(df.head())
df['TENURE'].value_counts()
x = df.iloc[:,[1,2,3,4]]
y = df.iloc[:,-1]
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['CUST_ID'] = le.fit_transform(df.CUST_ID.values)
pca2 = PCA(n_components=2)
principalComponents = pca2.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['TENURE']]], axis = 1)
finalDf.head()
from sklearn.cluster import KMeans
nclusters = 2 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)
# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)
from sklearn import metrics
score = metrics.silhouette_score(x, y_cluster_kmeans)
print(score)
scaler = StandardScaler()
X_Scale = scaler.fit_transform(x)
pca2 = PCA(n_components=2)
principalComponents = pca2.fit_transform(X_Scale)
principalDf1 = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf1 = pd.concat([principalDf1, df[['TENURE']]], axis = 1)
finalDf1.head()
from sklearn.cluster import KMeans
nclusters = 2 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(X_Scale)
# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_Scale)
from sklearn import metrics
score = metrics.silhouette_score(X_Scale, y_cluster_kmeans)
print(score)