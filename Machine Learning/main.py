import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score




##################################################Pré-processamento#####################################################

#seeds = pd.read_excel('seeds.xls')
seeds = pd.read_csv("/Users/mirellam./Desktop/Redes Neurais Não Supervisionadas/trab/seeds/seeds.txt", sep=r'\t',engine='python')
#print(seeds) #Display the whole dataset
#print(seeds.head())
X = seeds.iloc[:, 0:7].values #iloc function enables us to select a particular cell of the dataset, that is, it helps us select a value that belongs to a particular row or column from a set of values of a data frame or dataset.
XX = seeds.iloc[:, 0:8].values
dX = pd.DataFrame(X, columns = ['Area', 'Perimeter','Compactness','Kernel Length', 'Kernel Width', 'Assymetry Coefficient','Kernel Groove Length'])
dXX = pd.DataFrame(XX, columns = ['Area', 'Perimeter','Compactness','Kernel Length', 'Kernel Width', 'Assymetry Coefficient','Kernel Groove Length', 'Original Labels'])
dX.loc[-1] = [15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22]  # adding a row
dX.index = dX.index + 1  # shifting index
dX.sort_index(inplace=True)
dXX.loc[-1] = [15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22, 1]  # adding a row
dXX.index = dXX.index + 1  # shifting index
dXX.sort_index(inplace=True)
X = dX.values #transformar o dataframe na matriz correspondente
print("Original Dataframe:", dX)

print("Size:",dX.size)

print(dX.isnull().sum())

#Analisando os histogramas criados, podemos perceber que os atributos mostram uma distribuição multimodal

figure, axis = plt.subplots(5, 6, figsize = (40,40))

axis[0, 0].scatter(X[:, 1], X[:, 0]) #gráficos estão rotacionados em relação à imagem scatter-histogram2_seeds.png
axis[0, 0].set_ylabel("Área", fontsize=15)
axis[0, 0].set_xlabel('Perímetro', fontsize=15)



axis[0, 1].scatter(X[:, 2], X[:,0])
axis[0, 1].set_ylabel("Área", fontsize=15)
axis[0, 1].set_xlabel('Compacidade', fontsize=15)



axis[0, 2].scatter(X[:, 3], X[:, 0])
axis[0, 2].set_ylabel("Área", fontsize=15)
axis[0, 2].set_xlabel('Comprimento do Núcleo', fontsize=15)


axis[0, 3].scatter(X[:, 4], X[:, 0])
axis[0, 3].set_ylabel("Área", fontsize=15)
axis[0, 3].set_xlabel('Largura do Núcleo', fontsize=15)


axis[0, 4].scatter(X[:, 5], X[:, 0])
axis[0, 4].set_ylabel("Área", fontsize=15)
axis[0, 4].set_xlabel('Coeficiente de Assimetria', fontsize=15)


axis[0, 5].scatter(X[:, 6], X[:, 0])
axis[0, 5].set_ylabel("Área", fontsize=15)
axis[0, 5].set_xlabel('Comprimento do Sulco do Núcleo', fontsize=15)


axis[1, 0].scatter(X[:,2], X[:,1])
axis[1, 0].set_ylabel("Perímetro", fontsize=15)
axis[1, 0].set_xlabel('Compacidade', fontsize=15)



axis[1, 1].scatter(X[:,3], X[:,1])
axis[1, 1].set_ylabel("Perímetro", fontsize=15)
axis[1, 1].set_xlabel('Comprimento do Núcleo', fontsize=15)


axis[1, 2].scatter(X[:,4], X[:,1])
axis[1, 2].set_ylabel("Perímetro", fontsize=15)
axis[1, 2].set_xlabel('Largura do Núcleo', fontsize=15)


axis[1, 3].scatter(X[:,5], X[:,1])
axis[1, 3].set_ylabel("Perímetro", fontsize=15)
axis[1, 3].set_xlabel('Coeficiente de Assimetria', fontsize=15)


axis[1, 4].scatter(X[:,6], X[:,1])
axis[1, 4].set_ylabel("Perímetro", fontsize=15)
axis[1, 4].set_xlabel('Comprimento do Sulco do Núcleo', fontsize=15)



axis[2, 0].scatter(X[:,2], X[:,3])
axis[2, 0].set_ylabel('Comprimento do Núcleo', fontsize=15)
axis[2, 0].set_xlabel('Compacidade', fontsize=15)


axis[2, 1].scatter(X[:,4], X[:,3])
axis[2, 1].set_ylabel('Comprimento do Núcleo', fontsize=15)
axis[2, 1].set_xlabel('Largura do Núcleo', fontsize=15)


axis[2, 2].scatter(X[:,5], X[:,3])
axis[2, 2].set_ylabel("Comprimento do Núcleo", fontsize=15)
axis[2, 2].set_xlabel('Coeficiente de Assimetria', fontsize=15)


axis[2, 3].scatter(X[:,6], X[:,3])
axis[2, 3].set_ylabel("Comprimento do Núcleo", fontsize=15)
axis[2, 3].set_xlabel('Comprimento do Sulco do Núcleo', fontsize=15)


axis[3, 0].scatter(X[:,5], X[:,4])
axis[3, 0].set_ylabel("Largura do Núcleo", fontsize=15)
axis[3, 0].set_xlabel('Coeficiente de Assimetria', fontsize=15)


axis[3, 1].scatter(X[:,6], X[:,4])
axis[3, 1].set_ylabel("Largura do Núcleo do Grão", fontsize=15)
axis[3, 1].set_xlabel('Comprimento do Sulco do Núcleo', fontsize=15)


axis[4, 0].scatter(X[:,6], X[:,5])
axis[4, 0].set_ylabel('Coeficiente de Assimetria', fontsize=15)
axis[4, 0].set_xlabel('Largura do Núcleo', fontsize=15)


axis[1,5].set_axis_off()
axis[2,4].set_axis_off()
axis[2,5].set_axis_off()
axis[3,2].set_axis_off()
axis[3,3].set_axis_off()
axis[3,4].set_axis_off()
axis[3,5].set_axis_off()
axis[4,1].set_axis_off()
axis[4,2].set_axis_off()
axis[4,3].set_axis_off()
axis[4,4].set_axis_off()
axis[4,5].set_axis_off()

plt.savefig('scatter_seeds.png')
plt.show()

dX.rename(columns={'Area': 'Área', 'Perimeter': 'Perímetro', 'Compactness': 'Compacidade', 'Kernel Length':'Comprimento do Núcleo', 'Kernel Width':'Largura do Núcleo', 'Assymetry Coefficient':'Coeficiente de Assimetria', 'Kernel Groove Length':'Comprimento do Sulco do Núcleo'}, inplace=True)

#checando para ver se existem outliers
plt.figure(figsize=(15,15))
plt.suptitle('Boxplot Before Removing Outliers')
plt.subplot(4,4,1)
k1=sns.boxplot(dX['Área'], color = 'm')
plt.subplot(4,4,2)
k2=sns.boxplot(dX['Perímetro'], color = 'g')
plt.subplot(4,4,3)
k3=sns.boxplot(dX['Compacidade'], color = 'darkgray')
plt.subplot(4,4,4)
k4=sns.boxplot(dX['Comprimento do Núcleo'], color = 'r')
plt.subplot(4,4,5)
k5=sns.boxplot(dX['Largura do Núcleo'], color = 'pink')
plt.subplot(4,4,6)
k6=sns.boxplot(dX['Coeficiente de Assimetria'], color = "blue")
plt.subplot(4,4,7)
k7=sns.boxplot(dX['Comprimento do Sulco do Núcleo'], color = 'darkorange')
plt.show()

print("Highest allowed (Area)",dX['Área'].mean() + 2*dX['Área'].std())
print("Lowest allowed (Area)",dX['Área'].mean() - 2*dX['Área'].std())
print("Highest allowed (Perimeter)",dX['Perímetro'].mean() + 2*dX['Perímetro'].std())
print("Lowest allowed (Perimeter)",dX['Perímetro'].mean() - 2*dX['Perímetro'].std())
print("Highest allowed (Compactness)",dX['Compacidade'].mean() + 2*dX['Compacidade'].std())
print("Lowest allowed (Compactness)",dX['Compacidade'].mean() - 2*dX['Compacidade'].std())
print("Highest allowed (Kernel Length)",dX['Comprimento do Núcleo'].mean() + 2*dX['Comprimento do Núcleo'].std())
print("Lowest allowed (Kernel Length)",dX['Comprimento do Núcleo'].mean() - 2*dX['Comprimento do Núcleo'].std())
print("Highest allowed (Kernel Width)",dX['Largura do Núcleo'].mean() + 2*dX['Largura do Núcleo'].std())
print("Lowest allowed (Kernel Width)",dX['Largura do Núcleo'].mean() - 2*dX['Largura do Núcleo'].std())
print("Highest allowed (Assymetry Coefficient)",dX['Coeficiente de Assimetria'].mean() + 2*dX['Coeficiente de Assimetria'].std())
print("Lowest allowed (Assymetry Coefficient)",dX['Coeficiente de Assimetria'].mean() - 2*dX['Coeficiente de Assimetria'].std())
print("Highest allowed (Kernel Groove Length)",dX['Comprimento do Sulco do Núcleo'].mean() + 2*dX['Comprimento do Sulco do Núcleo'].std())
print("Lowest allowed (Kernel Groove Length)",dX['Comprimento do Sulco do Núcleo'].mean() - 2*dX['Comprimento do Sulco do Núcleo'].std())

#Trimming of Outliers
#dX_new = dX[(dX['Area'] < 20.666922670898536) & (dX['Area'] > 9.028124948149093)]
#dX_new = dX[(dX['Perimeter'] < 17.17120316741376) & (dX['Perimeter'] > 11.947368261157674)]
#dX_new = dX[(dX['Compactness'] < 0.9182574045962644) & (dX['Compactness'] > 0.8237397382608784)]
#dX_new = dX[(dX['Kernel Length'] < 6.513691743897599) & (dX['Kernel Length'] > 4.740841589435739)]
#dX_new = dX[(dX['Kernel Width'] < 4.019801691213609) & (dX['Kernel Width'] > 2.485836404024486)]
#dX_new = dX[(dX['Assymetry Coefficient'] < 6.729837935773989) & (dX['Assymetry Coefficient'] > 0.7033163499402946)]
#dX_new = dX[(dX['Kernel Groove Length'] < 6.416892302121196) & (dX['Kernel Groove Length'] > 4.380602935974043)]

plt.figure(figsize = (24,24))
subplot(4,4,1)
ax = sns.distplot(X[:,0], color="m")
plt.title('Área')
plt.xlabel('X[:,0]')
plt.ylabel("Densidade")

subplot(4,4,2)
ax = sns.distplot(X[:,1], color="g")
plt.title('Perímetro')
plt.xlabel('X[:,1]')
plt.ylabel("Densidade")

subplot(4,4,3)
ax = sns.distplot(X[:,2], color='darkgray')
plt.title('Compacidade')
plt.xlabel('X[:,2]')
plt.ylabel("Densidade")

subplot(4,4,4)
ax = sns.distplot(X[:,3], color="r")
plt.title('Comprimento do Núcleo')
plt.xlabel('X[:,3]')
plt.ylabel("Densidade")

subplot(4,4,5)
ax = sns.distplot(X[:,4], color="pink")
plt.title('Largura do Núcleo')
plt.xlabel('X[:,4]')
plt.ylabel("Densidade")

subplot(4,4,6)
ax = sns.distplot(X[:,5], color="blue")
plt.title('Coeficiente de Assimetria')
plt.xlabel('X[:,5]')
plt.ylabel("Densidade")

subplot(4,4,7)
ax = sns.distplot(X[:,6], color="darkorange")
plt.title('Comprimento do Sulco do Núcleo')
plt.xlabel('X[:,6]')
plt.ylabel("Densidade")

plt.suptitle('Probability Density Function')
plt.savefig('density_seeds1.png')
plt.show()

pearsoncorr = dX.corr(method='pearson')



#padronização dos dados
X_padronizado = preprocessing.scale(X)
plt.plot(X_padronizado, color = 'salmon')
plt.title('Scaled Seeds Dataset')
plt.savefig('scaled_seeds.png')
plt.show()
print("Variance:", np.var(X_padronizado)) #variância=1
print("Mean:", np.mean(X_padronizado)) #média=0
dX_padronizado = pd.DataFrame(X_padronizado, columns = ['Área', 'Perímetro','Compacidade','Comprimento do Núcleo', 'Largura do Núcleo', 'Coeficiente de Assimetria','Comprimento do Sulco do Núcleo'])


#cálculo das principais componentes dos dados
pca = PCA()
pca.fit_transform(X_padronizado)
plt.plot(range(0,len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_ * 100, 'o--', alpha=0.5, color='darkorange')
plt.title("PCA Explained Variance Ratio")
plt.xlabel('Componentes Principais')
plt.ylabel('Razão da Variância (%)')
plt.grid()
plt.savefig('pca_explained_variance_ratio.png')
plt.show()

Dd = pd.DataFrame(pca.components_, columns = ['Área', 'Perímetro','Compacidade','Comprimento do Núcleo', 'Largura do Núcleo', 'Coeficiente de Assimetria','Comprimento do Sulco do Núcleo'])
fig, ax = plt.subplots(figsize=(25,25))
sns.heatmap(Dd,
            xticklabels=Dd,
            yticklabels=['CP0', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6'],
            cbar_kws={"shrink": .8},
            annot=True,
            linewidth=0.5)
plt.title("Correlation Between Original Features and Principal Components")
plt.savefig('heatmap_pca.png')

print(Dd)




#The only reason to remove highly correlated features is storage and speed concerns.
pcaa = PCA(0.99)
X_new = pcaa.fit_transform(X_padronizado) #Principal component analysis (PCA) is a dimension reduction and decorrelation technique that transforms a correlated multivariate distribution into orthogonal linear combinations of the original variables
print("X_new.shape:",X_new.shape) #essa resposta quer dizer que preciso de 210 linhas 4 colunas/parâmetros para representar 99% do sinal
print("PCA Components:",pcaa.components_)
print("PCA Variance:",pcaa.explained_variance_)
print("Variance Ratio:", pcaa.explained_variance_ratio_ *100) #A 1º componente principal responde ou "explica" por 70.98% da variabilidade geral -> pca.explained_variance_/sum(pca.explained_variance_)
print("Sum of Variance Ratio of All Components:", pcaa.explained_variance_ratio_.sum()*100) #componentes 0, 1, 2 e 3 respondem pela maior parte (+99%) da variabilidade geral do conjunto de dados
sum = 70.96420217 + 16.83705092 + 9.92464479 + 1.57358305
print("Sum of Variance Ratio of the 1º, 2º, 3º and 4º Components:", sum)

dX_new = pd.DataFrame(X_new, columns = ['PC0', 'PC1','PC2','PC3'])
print("New Dataframe:", dX_new)

print("Variance:", np.var(X_new)) #variância=1.73
print("Mean:", np.mean(X_new)) #média=0

figure, axis = plt.subplots(3, 3, figsize = (18,18))

axis[0, 0].scatter(X_new[:, 1], X_new[:, 0]) #gráficos estão rotacionados em relação à imagem scatter-histogram2_seeds.png
axis[0, 0].set_ylabel("Componente Principal 0", fontsize=15)
axis[0, 0].set_xlabel('Componente Principal 1', fontsize=15)
plt.ylabel('Area')


axis[0, 1].scatter(X_new[:, 2], X_new[:,0])
axis[0, 1].set_ylabel("Componente Principal 0", fontsize=15)
axis[0, 1].set_xlabel('Componente Principal 2', fontsize=15)
plt.ylabel('Area')


axis[0, 2].scatter(X_new[:, 3], X_new[:, 0])
axis[0, 2].set_ylabel("Componente Principal 0", fontsize=15)
axis[0, 2].set_xlabel('Componente Principal 3', fontsize=15)
plt.ylabel('Area')

axis[1, 0].scatter(X_new[:,2], X_new[:,1])
axis[1, 0].set_ylabel("Componente Principal 1", fontsize=15)
axis[1, 0].set_xlabel('Componente Principal 2', fontsize=15)
plt.ylabel('Perimeter')

axis[1, 1].scatter(X_new[:,3], X_new[:,1])
axis[1, 1].set_ylabel("Componente Principal 1", fontsize=15)
axis[1, 1].set_xlabel('Componente Principal 3', fontsize=15)
plt.ylabel('Perimeter')

axis[2, 0].scatter(X_new[:,3], X_new[:,2])
axis[2, 0].set_ylabel("Componente Principal 2", fontsize=15)
axis[2, 0].set_xlabel('Componente Principal 3', fontsize=15)
plt.ylabel('Kernel Length')

axis[1,2].set_axis_off()
axis[2,1].set_axis_off()
axis[2,2].set_axis_off()

plt.show()
plt.savefig('scatter-histogram_seeds2.png')


##################################################Agrupamento###########################################################


#método elbow -> descobrir o número ótimo de classes

Fin = [] #dispersão intra classe

for i in range(1, 16):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(X_new)
    print(i, kmeans.inertia_) #Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.
    Fin.append(kmeans.inertia_) #adicionar kmeans.inertia à variável Fin

plt.plot(range(1, 16), Fin, 'o--', color='#FE036A', alpha=0.3)
plt.title('Natural Clusters')
plt.xlabel('Número de Classes')
plt.ylabel('Fin')  # within cluster sum of squares
plt.grid()
plt.savefig('n_clusters_seeds.png')
plt.show()

#No gráfico acima, há uma queda acentuada de Fin de n=1 para n=2, além disso, de n=3 (vizinhança imediata a uma grande variação de Fin) para outros n's não tem uma queda acentuada de Fin, então, pode-se concluir que o número ótimo de clusters é 3
d1 = 1459.7023696447482 - 657.9095488604958 #distancia do Fin de 1 ate 2 clusters
d2 = 657.9095488604958 - 434.8791832009574 #distancia do Fin de 2 ate 3 clusters
d3 = 434.8791832009574 - 375.21339050011005 #distancia do Fin de 3 ate 4 clusters
d4 = 375.21339050011005 - 328.2563691751317 #distancia do Fin de 4 ate 5 clusters
print("d1-d2=", d1)
print("d2-d3=", d2)
print("d3-d4=", d3)
print("d4-d5=", d4)


#Método da Silhueta para encontrar clusters ideais no agrupamento de K-Means. O método de silhueta calcula os coeficientes de silhueta de cada ponto que mede o quanto um ponto é semelhante ao seu próprio cluster em comparação com outros clusters.
#The silhouette ranges from −1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters
#for i in range(2, 10):
#    kmeans = KMeans(n_clusters=i,init='k-means++')
#    kmeans.fit_predict(X_new)
#    score = silhouette_score(X_new, kmeans.labels_, metric='euclidean')
#    print(score)


#aplicando k-means com um número de classes=3
kmeans = KMeans(n_clusters = 3, init = 'k-means++',max_iter=400, random_state=3)
kmeans.fit(X_new) #executar o método fit() para executar o algoritmo e agrupar os dados. O método fit() recebe como parâmetro os dados a serem agrupados, nesse caso será a variável X_new que declaramos anteriormente.
#Neste momento já temos os dados agrupados e vamos verificar os centroides gerados através do atributo cluster_centers_.
print("Centers:", kmeans.cluster_centers_)
print("Iterations:", kmeans.max_iter)

#cada instância contém três valores, e cada valor corresponde exatamente a distância entre a instância de dados corrente e cada um dos três clusters
distance = kmeans.fit_transform(X_new) #O método fit_transform() executa o K-means para agrupar os dados e retorna uma tabela de distâncias. A tabela de distâncias é criada de forma que em cada instância contém os valores de distância em relação a cada cluster.
print("Distance:",distance)

#Podemos conferir visualizando o atributo labels_ que nos retorna os labels para cada instância, ou seja, o código do cluster que a instância de dados foi atribuído.
labels = kmeans.labels_
print("labels:",labels)
print ("Número de elementos por cluster:", np.bincount(labels))

#visualizar através de uma representação gráfica os nossos dados e os centroides criados pelo K-means
figure, axis = plt.subplots(3, 3, figsize = (18,18))

axis[0, 0].scatter(X_new[:, 1], X_new[:, 0], s = 60, c = kmeans.labels_) #gráficos estão rotacionados em relação à imagem scatter-histogram2_seeds.png
axis[0, 0].scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s = 300, c = 'red',label = 'Centróides')
axis[0, 0].set_ylabel("Componente Principal 0", fontsize=15)
axis[0, 0].set_xlabel('Componente Principal 1', fontsize=15)
axis[0, 0].legend()
plt.ylabel('Area')


axis[0, 1].scatter(X_new[:, 2], X_new[:,0], s = 60, c = kmeans.labels_)
axis[0, 1].scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 0], s = 300, c = 'red',label = 'Centróides')
axis[0, 1].set_ylabel("Componente Principal 0", fontsize=15)
axis[0, 1].set_xlabel('Componente Principal 2', fontsize=15)
axis[0, 1].legend()
plt.ylabel('Area')


axis[0, 2].scatter(X_new[:, 3], X_new[:, 0], s = 60, c = kmeans.labels_)
axis[0, 2].scatter(kmeans.cluster_centers_[:, 3], kmeans.cluster_centers_[:, 0], s = 300, c = 'red',label = 'Centróides')
axis[0, 2].set_ylabel("Componente Principal 0", fontsize=15)
axis[0, 2].set_xlabel('Componente Principal 3', fontsize=15)
axis[0, 2].legend()
plt.ylabel('Area')


axis[1, 0].scatter(X_new[:,2], X_new[:,1], s = 60, c = kmeans.labels_)
axis[1, 0].scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 1], s = 300, c = 'red',label = 'Centróides')
axis[1, 0].set_ylabel("Componente Principal 1", fontsize=15)
axis[1, 0].set_xlabel('Componente Principal 2', fontsize=15)
axis[1, 0].legend()
plt.ylabel('Perimeter')


axis[1, 1].scatter(X_new[:,3], X_new[:,1], s = 60, c = kmeans.labels_)
axis[1, 1].scatter(kmeans.cluster_centers_[:, 3], kmeans.cluster_centers_[:, 1], s = 300, c = 'red',label = 'Centróides')
axis[1, 1].set_ylabel("Componente Principal 1", fontsize=15)
axis[1, 1].set_xlabel('Componente Principal 3', fontsize=15)
axis[1, 1].legend()
plt.ylabel('Perimeter')


axis[2, 0].scatter(X_new[:,3], X_new[:,2], s = 60, c = kmeans.labels_)
axis[2, 0].scatter(kmeans.cluster_centers_[:, 3], kmeans.cluster_centers_[:, 2], s = 300, c = 'red',label = 'Centróides')
axis[2, 0].set_ylabel('Componente Principal 2', fontsize=15)
axis[2, 0].set_xlabel('Componente Principal 3', fontsize=15)
axis[2, 0].legend()
plt.ylabel('Kernel Length')

axis[1,2].set_axis_off()
axis[2,1].set_axis_off()
axis[2,2].set_axis_off()

plt.suptitle('K-means Clusters and Centroids', fontsize=23)
plt.savefig('kmeans_final.png')
plt.show()




unique_labels, counts = np.unique(labels, return_counts=True)
label_order = np.argsort(counts)[::-1]  # descending order
unique_labels = unique_labels[label_order]
counts = counts[label_order]
percentages = counts / counts.sum() * 100
color_labels = [f'{label}:\n{perc:.1f} %' for label, perc in zip(unique_labels, percentages)]
colors = ['gold', 'red', 'lightblue']
plt.pie(counts, labels=color_labels, colors=colors)
plt.title('Elements per Cluster')
plt.savefig('pie_clusters.png')
plt.show()



##################################################Classificador#########################################################

#mapping = {0:'Rosa', 1:'Kama', 2:'Canadian'}
mappingm = {0:0, 1:1, 2:2}
labels = [mappingm[i] for i in labels]

#mapping = {1:'Kama', 2:'Rosa', 3:'Canadian'}
mapping = {1:1, 2:0, 3:2}
original_labels = [mapping[i] for i in dXX['Original Labels']]

# Using 'Labels' as the column name
# and equating it to the list
dX_newL = pd.DataFrame(X_new, columns = ['PC0', 'PC1','PC2','PC3'])
dX_newL['Labels'] = labels
print("dX_newL:", dX_newL)

x = dX_new
y = dX_newL['Labels']
y_original = original_labels




#70% training and 30% test - técnica de hold-out
#Now that our data set has been split into training data and test data, we’re ready to start training our model
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3, random_state=1)
x_training_data1, x_test_data1, y_training_data1, y_test_data1 = train_test_split(x, y_original, test_size = 0.3, random_state=1)
#x_training_data: variável associada aos dados de treino, y_training_data: variável associada às classes de treino
#x_test_data: variável associada aos dados de teste, y_test_data: variável associada às classes de teste



# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=6)

# Train Decision Tree Classifier
clf = clf.fit(x_training_data, y_training_data)

#Predict the response for test dataset
y_pred = clf.predict(x_test_data)
print("Accuracy:",metrics.accuracy_score(y_test_data, y_pred))


##################################################Pós-Processamento#####################################################


print(classification_report(y_test_data, y_pred))

conf_matrix = confusion_matrix(y_test_data1, y_pred)

ax= plt.subplot()
sns.heatmap(conf_matrix, annot=True, fmt='g', ax=ax, cmap='Purples')  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Classes Preditas');ax.set_ylabel('Classes Verdadeiras')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['0', '1', '2']); ax.yaxis.set_ticklabels(['0', '1', '2'])
plt.savefig('confusion_matrix.png')
plt.show()

val_scores = cross_val_score(clf, x_training_data, y_training_data, cv=5) #cv=18 gera 18 modelos de dados de treino e teste e avalia a acurácia de cada um
print('Acurácia nos k-folds:', val_scores)
print('Média: {:.2} | Desvio: {:.2}'.format(np.mean(val_scores), np.std(val_scores)))

#############CAMADA DE KOHONEN###############

nc = 3         # number of classes
W = []         # list for w vectors
M = len(X_new)     # number of x vectors
N = len(X_new[0])  # dimensionality of x vectors

#create a function for obtaining random values for the x vectors (or weights), and then we initialize these x vectors
def get_weights():
    y = np.random.random() * (2.0 / np.sqrt(M))
    return 0.5 - (1 / np.sqrt(M)) + y

for i in range(nc):
    W.append(list())
    for j in range(N):
        W[i].append(get_weights() * 0.5)

print("W[i]:", W)

#create a function for computing the Euclidian distance between our x and w vectors
def distance(w, x):
    r = 0
    for i in range(len(w)):
        r = r + (w[i] - x[i]) * (w[i] - x[i])

    r = np.sqrt(r)
    return r

#create a function for finding the closest x vectors to the w vectors
def Findclosest(W, x):
    wm = W[0]
    r = distance(wm, x)

    i = 0
    i_n = i

    for w in W:
        if distance(w, x) < r:
            r = distance(w, x)
            wm = w
            i_n = i
        i = i + 1

    return (wm, i_n)

#initialize the λ and Δλ coefficients and start looping to find the closest x vectors to the w vectors
la = 0.2    # λ coefficient
dla = 0.05  # Δλ

while la >= 0:
    for k in range(10):
        for x in X_new:
            wm = Findclosest(W, x)[0]
            for i in range(len(wm)):
                wm[i] = wm[i] + la * (x[i] - wm[i])

    la = la - dla

#network is trained. Finally, we can compare the results of our classification with the actual values from the data frame
Data = list()

for i in range(len(W)):
    Data.append(list())

dfList = original_labels

labels = []
DS = list()
i = 0
for x in X_new:
    i_n = Findclosest(W, x)[1]
    Data[i_n].append(x)
    DS.append([i_n, dfList[i]])
    labels.append(i_n)
    i = i + 1


print(DS)
print(labels)
print("Número de elementos por classe:", np.bincount(labels))

W = np.array(W)
print("centroids:",W)

print(labels[81])
print(labels[82])
print(labels[83])
print(labels[84])
print("85:",labels[85])
print(labels[86])
print(labels[87])
print(labels[88])
print(labels[89])

g = 'Kama'
gg = 'Rosa'
ggg = 'Canadian'

for i in range(len(labels)):
        if labels[i] == labels[0]:
            l = g
            lal = labels[i]
        if labels[i] == labels[209]:
            lll = ggg
            lalll = labels[i]
        if labels[i] == labels[85]:
            ll = gg
            lall = labels[i]

for i in range(len(labels)):
    if labels[i] == lal:
            labels[i] = l
    if labels[i] == lalll:
            labels[i] = lll
    if labels[i] == lall:
            labels[i] = ll

print(labels)

mappingm = {'Kama':3, 'Rosa':4, 'Canadian':5}
labels = [mappingm[i] for i in labels]

mapping = {1:3, 2:4, 3:5}
original_labels = [mapping[i] for i in dXX['Original Labels']]

print(np.bincount(labels))

print(labels)

#visualizar através de uma representação gráfica os nossos dados e os centroides criados pelo K-means
figure, axis = plt.subplots(3, 3, figsize = (18,18))

axis[0, 0].scatter(X_new[:, 1], X_new[:, 0], s = 60, c = labels) #gráficos estão rotacionados em relação à imagem scatter-histogram2_seeds.png
axis[0, 0].scatter(W[:, 1], W[:, 0], s = 300, c = 'red',label = 'Centróides')
axis[0, 0].set_ylabel("Componente Principal 0", fontsize=15)
axis[0, 0].set_xlabel('Componente Principal 1', fontsize=15)
axis[0, 0].legend()
plt.ylabel('Area')


axis[0, 1].scatter(X_new[:, 2], X_new[:,0], s = 60, c = labels)
axis[0, 1].scatter(W[:, 2], W[:, 0], s = 300, c = 'red',label = 'Centróides')
axis[0, 1].set_ylabel("Componente Principal 0", fontsize=15)
axis[0, 1].set_xlabel('Componente Principal 2', fontsize=15)
axis[0, 1].legend()
plt.ylabel('Area')


axis[0, 2].scatter(X_new[:, 3], X_new[:, 0], s = 60, c = labels)
axis[0, 2].scatter(W[:, 3], W[:, 0], s = 300, c = 'red',label = 'Centróides')
axis[0, 2].set_ylabel("Componente Principal 0", fontsize=15)
axis[0, 2].set_xlabel('Componente Principal 3', fontsize=15)
axis[0, 2].legend()
plt.ylabel('Area')


axis[1, 0].scatter(X_new[:,2], X_new[:,1], s = 60, c = kmeans.labels_)
axis[1, 0].scatter(W[:, 2], W[:, 1], s = 300, c = 'red',label = 'Centróides')
axis[1, 0].set_ylabel("Componente Principal 1", fontsize=15)
axis[1, 0].set_xlabel('Componente Principal 2', fontsize=15)
axis[1, 0].legend()
plt.ylabel('Perimeter')


axis[1, 1].scatter(X_new[:,3], X_new[:,1], s = 60, c = labels)
axis[1, 1].scatter(W[:, 3], W[:, 1], s = 300, c = 'red',label = 'Centróides')
axis[1, 1].set_ylabel("Componente Principal 1", fontsize=15)
axis[1, 1].set_xlabel('Componente Principal 3', fontsize=15)
axis[1, 1].legend()
plt.ylabel('Perimeter')


axis[2, 0].scatter(X_new[:,3], X_new[:,2], s = 60, c = labels)
axis[2, 0].scatter(W[:, 3], W[:, 2], s = 300, c = 'red',label = 'Centróides')
axis[2, 0].set_ylabel('Componente Principal 2', fontsize=15)
axis[2, 0].set_xlabel('Componente Principal 3', fontsize=15)
axis[2, 0].legend()
plt.ylabel('Kernel Length')

axis[1,2].set_axis_off()
axis[2,1].set_axis_off()
axis[2,2].set_axis_off()

plt.suptitle('Kohonen Clusters and Centroids', fontsize=23)
plt.savefig('kohonen_final.png')
plt.show()

unique_labels, counts = np.unique(labels, return_counts=True)
label_order = np.argsort(counts)[::-1]  # descending order
unique_labels = unique_labels[label_order]
counts = counts[label_order]
percentages = counts / counts.sum() * 100
color_labels = [f'{label}:\n{perc:.1f} %' for label, perc in zip(unique_labels, percentages)]
colors = ['pink', 'cornflowerblue', 'lightgreen']
plt.pie(counts, labels=color_labels, colors=colors)
#dX_new['Labels'].plot(kind = "bar")
plt.title('Elements per Cluster')
plt.savefig('pie_clusters_kohonen.png')
plt.show()


#Classificador
# Using 'Labels' as the column name
# and equating it to the list
dX_newL = pd.DataFrame(X_new, columns = ['CP0', 'CP1','CP2','CP3'])
dX_newL['Labels'] = labels
print("dX_newL:", dX_newL)

x = dX_new
y = dX_newL['Labels']
y_original = original_labels

# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# define lists to collect scores
train_scores, test_scores = list(), list()
# define the tree depths to evaluate
values = [i for i in range(1, 51)]
# evaluate a decision tree for each depth
for i in values:
	# configure the model
	model = DecisionTreeClassifier(max_depth=i)
	# fit model on the training dataset
	model.fit(X_train, y_train)
	# evaluate on the train dataset
	train_yhat = model.predict(X_train)
	train_acc = accuracy_score(y_train, train_yhat)
	train_scores.append(train_acc)
	# evaluate on the test dataset
	test_yhat = model.predict(X_test)
	test_acc = accuracy_score(y_test, test_yhat)
	test_scores.append(test_acc)
	# summarize progress
	print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
# plot of train and test scores vs number of neighbors
plt.plot(values, train_scores, '-o', label='Train')
plt.plot(values, test_scores, '-o', label='Test')
plt.xlabel('Máxima Profundidade')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

#70% training and 30% test - técnica de hold-out
#Now that our data set has been split into training data and test data, we’re ready to start training our model
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3, random_state=1)
x_training_data1, x_test_data1, y_training_data1, y_test_data1 = train_test_split(x, y_original, test_size = 0.3, random_state=1)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=6)

# Train Decision Tree Classifier
clf = clf.fit(x_training_data, y_training_data)

#Predict the response for test dataset
y_pred = clf.predict(x_test_data)
print("Accuracy:",metrics.accuracy_score(y_test_data, y_pred))

#Pós-Processamento

print(classification_report(y_test_data, y_pred))

conf_matrix = confusion_matrix(y_test_data1, y_pred)

ax= plt.subplot()
sns.heatmap(conf_matrix, annot=True, fmt='g', ax=ax, cmap='Greens')  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Classes Preditas');ax.set_ylabel('Classes Verdadeiras')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['3', '4', '5']); ax.yaxis.set_ticklabels(['3', '4', '5'])
plt.savefig('confusion_matrix_kohonen.png')
plt.show()

val_scores = cross_val_score(clf, x_training_data, y_training_data, cv=5) #cv=5 gera 5 modelos de dados de treino e teste e avalia a acurácia de cada um
print('Acurácia nos k-folds:', val_scores)
print('Média: {:.2} | Desvio: {:.2}'.format(np.mean(val_scores), np.std(val_scores)))

fig, ax = plt.subplots(figsize=(25,25))
sns.heatmap(pearsoncorr,
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cbar_kws={"shrink": .8},
            annot=True,
            linewidth=0.5)

plt.title("Pearson Correlation Between Features")
plt.savefig('heatmap_seeds.png')
plt.show()




