import tp2_aux_altered
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, Isomap
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import f_classif
from pandas.plotting import scatter_matrix
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import rand_score, adjusted_rand_score
import math
import os.path
from ClusterIndicators import ClusterIndicators
from sklearn.cluster import AffinityPropagation


#TODO:
# - Responder Perguntas
# - Algoritmo extra
# - Dar plot com labels em 2 dimensoes

def sfs(cluster_algorithm, highestIndexesF, labels, verbose=False):
    
    curr_combination = np.array([])
    rand_last_iter = -math.inf
    
    #Iniciar a variavel que determinamos como booleana mas que no python é determinada como um inteiro de 2 bytes a 1, ou seja, true (verdadeiro)
    cont = 1
    max_features = 5
    
    labels = labels.flatten()
    
    while cont:
        
        best_rand = -math.inf
        best_index = 0
        
        #If we have the desired number of features, end
        if len(curr_combination) == max_features:
            break
        
        #Try every feature combined with the previous best combination
        for i in highestIndexesF:
            
            index_to_use = [i]
            index_to_use = np.append(index_to_use , curr_combination)
            # print("TRYING:", index_to_use)
            data_aux = df.iloc[:,index_to_use]
            
            #iterar sobre os eps sem dar erro na sillhoutte score
            clusters = cluster_algorithm.fit_predict(data_aux)
            # dbscan = DBSCAN(eps = elbow, min_samples=5).fit_predict(data_aux)
            
            a_rand_score = adjusted_rand_score(labels[labels != 0], clusters[labels != 0])
            # silhouette_score_dbscan = silhouette_score(data_aux, dbscan)
            
            print("Rand score: ", a_rand_score, "\n", index_to_use) if verbose else 0
            
            if a_rand_score > best_rand:
                best_rand = a_rand_score
                best_index = i
              
        if rand_last_iter > best_rand:
            break
        
        #Add best feature of this iteration
        curr_combination = np.append(curr_combination , [best_index])
        
        #Remove feature from features that can be added
        index_to_del = highestIndexesF.tolist().index(best_index)
        highestIndexesF = np.delete(highestIndexesF, index_to_del)
        
        #Update last silhouette score (useful if we want to stop when sil score lowers)
        rand_last_iter = best_rand
    
    return curr_combination


imagesMatrix = tp2_aux_altered.images_as_matrix()

############## PCA ############

#With each method, extract six features from the data set, for a total of 18 features.
#TODO Try standardizing first
pca = PCA(6)
fitTransformImagesPCA = pca.fit_transform(imagesMatrix)


#https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
scaler = StandardScaler()
scaledTransformedDataPCA = scaler.fit_transform(fitTransformImagesPCA)

loadLabels = np.loadtxt("labels.txt", delimiter = ",")
#ignorar todas as linhas com 0 na classe (nao esta definida) e guardar as labels das restantes
#fazer reshape para o por numa coluna para depois dar append aos dados respetivos

labels = np.reshape(loadLabels[:,-1], (len(loadLabels[:,-1]),1))

#dar append das labels a informacao respetiva
pca_array = np.append(scaledTransformedDataPCA, labels, axis=1)

############################## VISUALIZACAO  #####################
#Parallel Coordinates
df_pca = pd.DataFrame(pca_array,columns = ['pca_1','pca_2','pca_3','pca_4','pca_5','pca_6','labels'])

#Parallel Coodinates
#correlacao entre todas
parallel_coordinates(df_pca,'labels', color= ('r','g','b'))
plt.savefig("plots/visualization/pca_pc.png", dpi = 256)
plt.show()
plt.close()

#correlacao duas a duas features
scatter_matrix(df_pca.iloc[:,[0,1,2,3,4,5]], alpha=0.5,figsize=(15,10), diagonal="kde")
plt.savefig("plots/visualization/pca_sm.png", dpi = 256)
plt.show()
plt.close()
####################################################################################################

                                    ############## TSNE ############

fetch_from_file = 1
if not os.path.isfile('./tsne.csv') and fetch_from_file:
    tsne = TSNE(6, method = 'exact')
    fitTransformImagesTSNE = tsne.fit_transform(imagesMatrix)
    scaledTransformedDataTSNE = scaler.fit_transform(fitTransformImagesTSNE)
    
    np.savetxt('tsne.csv', scaledTransformedDataTSNE, delimiter=',')
    
else:
    scaledTransformedDataTSNE = np.loadtxt('tsne.csv', delimiter=',')


tsne_array = np.append(scaledTransformedDataTSNE, labels, axis=1)



############################# VISUALIZACAO #####################
df_tsne = pd.DataFrame(tsne_array,columns = ['tsne_1','tsne_2','tsne_3','tsne_4','tsne_5','tsne_6','labels'])

#Parallel Coodinates
parallel_coordinates(df_tsne,'labels', color= ('r','g','b'))
plt.savefig("plots/visualization/tsne_pc.png", dpi = 256)
plt.show()
plt.close()

scatter_matrix(df_pca.iloc[:,[0,1,2,3,4,5]], alpha=0.5,figsize=(15,10), diagonal="kde")
plt.savefig("plots/visualization/tsne_sm.png", dpi = 256)
plt.show()
plt.close()

####################################################################################################

                                    ############## ISOMAP ############

isomap = Isomap(n_components=6)
fitTransformImagesISO = isomap.fit_transform(imagesMatrix)
scaledTransformedDataISO = scaler.fit_transform(fitTransformImagesISO)     

isomap_array = np.append(scaledTransformedDataISO, labels, axis=1)


############################### VISUALIZACAO #####################

df_iso = pd.DataFrame(isomap_array,columns = ['iso_1','iso_2','iso_3','iso_4','iso_5','iso_6','labels'])

#Parallel Coodinates
parallel_coordinates(df_iso,'labels', color= ('r','g','b'))
plt.savefig("plots/visualization/isomap_pc.png", dpi = 256)
plt.show()
plt.close()

scatter_matrix(df_pca.iloc[:,[0,1,2,3,4,5]], alpha=0.5,figsize=(15,10), diagonal="kde")
plt.savefig("plots/visualization/isomap_sm.png", dpi = 256)
plt.show()
plt.close()
####################################################################################################


############################################ FEATURE SELECTION #####################################
                                        ######### DBSCAN #########
                                        
feat_array = np.concatenate((scaledTransformedDataPCA, \
                             scaledTransformedDataTSNE,scaledTransformedDataISO),axis=1)
df = pd.DataFrame(feat_array)

#escolher features para se passar aqui
#ANOVA Compara a variancia entre grupos com a variancia dentro dos grupos.
#Se o valor for alto, a variancia dessas features é mais alta e retêm mais informacao por isso queremos guardar estas.

#-------------------Univariate Filtering-------------------
labeled_feat_array = []
for i in range(0, len(labels)):
    l = labels[i]
    if l:
        labeled_feat_array.append(feat_array[i])
        
labeled_feat_array = np.array(labeled_feat_array)


f,prob = f_classif(labeled_feat_array, labels[labels != 0])
# print(str(f) + "====" + str(prob))

#guardar os indexes dos melhores Fs (corerspondem aos atributos p guardar)
#https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array

num_feats = 10 #Number of features
highestIndexesF = (-f).argsort()[:num_feats]
print("10 Highest Indexes, according to ANOVA: ", highestIndexesF)
#----------------------------------------------------------


#-------------------Multivariate Filtering-----------------


elbow = 1.25

labels = labels.flatten()

highestIndexesF_DBSCAN = sfs(DBSCAN(eps = elbow, min_samples=5), highestIndexesF.copy(), labels)
print("Selected features for DBSCAN:", highestIndexesF_DBSCAN)

highestIndexesF_KMEANS = sfs(KMeans(n_clusters = 4, random_state = 5), highestIndexesF.copy(), labels)
print("Selected features for KMEANS:", highestIndexesF_KMEANS)

highestIndexesF_AP = sfs(AffinityPropagation(damping = 0.91, random_state = 5,\
                                             max_iter = 400, convergence_iter = 15 ), \
                                             highestIndexesF.copy(), labels)
print("Selected features for Affinity Propagation:", highestIndexesF_AP)



#----------------------------------------------------------

#--------Ploting 2D Cluster--------------------------------
data_2d = df.iloc[:,highestIndexesF[0:1]]

plt.plot(data_2d, '.')
plt.title("2D Visualization of Cluster")
plt.xlabel("Feature " + str(highestIndexesF[0]))
plt.ylabel("Feature " + str(highestIndexesF[1]))
plt.savefig('plots/2D_Cluster.png', dpi = 256)
plt.show()
plt.close()
#----------------------------------------------------------


#fazer a ordenacao dos vizinhos
#ir buscar a distancia do quinto vizinho mais perto
vectorOnes = np.ones(len(df.iloc[:,highestIndexesF]))
data_DBSCAN = df.iloc[:,highestIndexesF_DBSCAN]
data_KMEANS = df.iloc[:,highestIndexesF_KMEANS]
data_AP = df.iloc[:,highestIndexesF_AP]

#ver aqui se so comparamos com os que temos classe ou nao
knn = KNeighborsClassifier(n_neighbors = 5).fit(data_DBSCAN, vectorOnes)
dist = knn.kneighbors(n_neighbors = 5)

#a estrutura tem duas coisas, a primeira é a das distancias (segunda o tipo de dados), pegar a coluna dos quintos vizinhos mais proximos
dists = np.sort(dist[0][:, 4])
#ordenar as distancias aos quintos vizinhos
dists_sort = dists[::-1]

#plt.plot(np.sort(indices[:,3]),dists_sort,label = 'Distance')
plt.plot(dists_sort, '-r' ,label = 'Distance')
#ir buscar a distancia ao quinto vizinho
plt.scatter(25, elbow, label = 'eps ideal (elbow) = ' + str(elbow))
plt.legend()
plt.title("5th Neighbor Distance || IDs")
plt.xlabel("IDs")
plt.ylabel("5th Neighbor Distance")
plt.savefig('plots/elbow.png', dpi = 256)
plt.show()
plt.close()

#min samples a 5 como está no enunciado (contrario ao paper)
#DBSCAN
best_dbscan_elbow = DBSCAN(eps = elbow,min_samples=5).fit_predict(data_DBSCAN)
tp2_aux_altered.report_clusters(loadLabels[:,0],best_dbscan_elbow, \
                        'html/dbscan_clusters_eps_elbow=' + str(elbow) + '.html')


#KMEANS
#4 clusters = 3 para as 3 fases e 1 para as noise (apriori)
num_of_clusters = 2
kmeansApriori = KMeans(n_clusters = num_of_clusters, random_state = 5).fit_predict(data_KMEANS)
tp2_aux_altered.report_clusters(loadLabels[:,0],kmeansApriori,'html/kmeans_' + str(num_of_clusters) + 'clusters.html')

#TESTAR METRICAS DBSCAN
dbscanMetrics = pd.DataFrame(columns = ['eps', 'randIndex', 'precision', \
                                        'recall', 'f1', 'adjustedRandIndex', 'silhouette'])

min_range = 0.7
max_range = 2.5
step = 0.05
epsRange = np.arange(min_range, max_range + 0.1, step)

# labels = labels.flatten()
for eps in epsRange:    

    #iterar sobre os eps sem dar erro na sillhoutte score
    dbscan = DBSCAN(eps = eps, min_samples=5).fit_predict(data_DBSCAN)
    
    randIndex = rand_score(labels[labels != 0], dbscan[labels != 0])    
    
    ci = ClusterIndicators(dbscan[labels!=0], labels[labels != 0])
    precision = ci.precision
    recall = ci.recall
    f1 = ci.f1

    
    adjustedRandScore = adjusted_rand_score(labels[labels != 0], dbscan[labels != 0])

    label_1 = dbscan[0]
    if(len(data_DBSCAN) == len(dbscan[dbscan == label_1])):
        #silhouette score throws exception when all points are labeled the same
        #CLEVER trick to get around this and calculate silhouette score
        dbscan[0] = 1
        
    silhouette_score_dbscan = silhouette_score(data_DBSCAN, dbscan)

    #passar o eps certo
    dbscanMetrics.loc[eps] = eps, randIndex, precision, recall, f1,\
    adjustedRandScore, silhouette_score_dbscan


#ver qual o eps cujo valor da silhueta e o mais alto
indexMaxSilhouette = dbscanMetrics["silhouette"].idxmax()
maxSilhouette = dbscanMetrics["silhouette"].max()

print("DBSCAN - Max Silhouette: ", maxSilhouette)
print("DBSCAN - Max Adjusted Rand Index", dbscanMetrics["adjustedRandIndex"].max())
print("DBSCAN - Max Precision", dbscanMetrics["precision"].max())
print("DBSCAN - Max Recall", dbscanMetrics["recall"].max())
print("DBSCAN - Max F1", dbscanMetrics["f1"].max())
print("DBSCAN - Max Rand Index", dbscanMetrics["randIndex"].max())

bestEps = round(dbscanMetrics["eps"].loc[indexMaxSilhouette], 3)
print("DBSCAN - Best Eps: ", bestEps, "\n")
dbscanOtimizado = DBSCAN(eps = bestEps, min_samples=5).fit_predict(data_DBSCAN)
tp2_aux_altered.report_clusters(loadLabels[:,0], dbscanOtimizado, 'html/dbscan_bestEps=' + str(bestEps) +'.html')


plt.plot(dbscanMetrics.iloc[:,0],dbscanMetrics.iloc[:,1],label = 'Rand Index')
plt.plot(dbscanMetrics.iloc[:,0],dbscanMetrics.iloc[:,2],label = 'Precision')
plt.plot(dbscanMetrics.iloc[:,0],dbscanMetrics.iloc[:,3],label = 'Recall')
plt.plot(dbscanMetrics.iloc[:,0],dbscanMetrics.iloc[:,4],label = 'F1')
plt.plot(dbscanMetrics.iloc[:,0],dbscanMetrics.iloc[:,5],label = 'Adjusted Rand Index')
plt.plot(dbscanMetrics.iloc[:,0],dbscanMetrics.iloc[:,6],label = 'Silhouette')
plt.scatter(bestEps, maxSilhouette, color = 'red', label = 'eps chosen')
plt.legend(loc = 'upper left')
plt.xlabel('eps values')
plt.ylabel('Indexes Values')
plt.title('Indexes Values vs Number of Clusters')
plt.ylim(0, 1.1)
plt.xlim(min_range, max_range)
plt.savefig('plots/DBSCAN.png',dpi = 256)
plt.show()
plt.close()


##############################
kmeansMetrics = pd.DataFrame(columns = ['k', 'randIndex', 'precision', 'recall',\
                                        'f1', 'adjustedRandIndex', 'silhouette'])

min_k = 2
max_k = 15
kRange = np.arange(min_k, max_k + 0.1, 1)

labels = labels.flatten()
for k in kRange:
    
    k = int(k)
    kmeans = KMeans(n_clusters = k, random_state = 5).fit_predict(data_KMEANS)
    
    randIndex = rand_score(labels[labels != 0], kmeans[labels != 0])
    ci = ClusterIndicators(kmeans[labels!=0], labels[labels != 0])
    precision = ci.precision
    recall = ci.recall
    f1 = ci.f1

    silhouette_score_kmeans = silhouette_score(data_KMEANS, kmeans)
    
    adjustedRandScore = adjusted_rand_score(labels[labels != 0], kmeans[labels != 0])

    kmeansMetrics.loc[k] = k, randIndex, precision, recall, f1, adjustedRandScore, silhouette_score_kmeans


#ver qual o eps cujo valor da silhueta e o mais alto
indexMaxSilhouette = kmeansMetrics["silhouette"].idxmax()
maxSilhouette = kmeansMetrics["silhouette"].max()

print("KMeans - Max Silhouette: ", maxSilhouette)
print("KMeans - Max Adjusted Rand Index", kmeansMetrics["adjustedRandIndex"].max())
print("KMeans - Max Precision", kmeansMetrics["precision"].max())
print("KMeans - Max Recall", kmeansMetrics["recall"].max())
print("KMeans - Max F1", kmeansMetrics["f1"].max())
print("KMeans - Max Rand Index", kmeansMetrics["randIndex"].max())

bestK = int(kmeansMetrics["k"].loc[indexMaxSilhouette])
print("KMeans - Best K: ", bestK, "\n")

kmeansOtimizado = KMeans(n_clusters = bestK, random_state = 5).fit_predict(data_KMEANS)
tp2_aux_altered.report_clusters(loadLabels[:,0], kmeansOtimizado, 'html/kmeans_bestK=' + str(bestK) +'.html')

plt.plot(kmeansMetrics.iloc[:,0],kmeansMetrics.iloc[:,1],label = 'Rand Index')
plt.plot(kmeansMetrics.iloc[:,0],kmeansMetrics.iloc[:,2],label = 'Precision')
plt.plot(kmeansMetrics.iloc[:,0],kmeansMetrics.iloc[:,3],label = 'Recall')
plt.plot(kmeansMetrics.iloc[:,0],kmeansMetrics.iloc[:,4],label = 'F1')
plt.plot(kmeansMetrics.iloc[:,0],kmeansMetrics.iloc[:,5],label = 'Adjusted Rand Index')
plt.plot(kmeansMetrics.iloc[:,0],kmeansMetrics.iloc[:,6],label = 'Silhouette')
plt.scatter(bestK, maxSilhouette, color = 'red', label = 'k chosen = ' + str(bestK))
plt.legend(loc = 'upper left')
plt.xlabel('k values')
plt.ylabel('Indexes Values')
plt.title('Indexes Values vs Number of Clusters')
plt.ylim(0, 1.1)
plt.xlim(min_k - 0.1, max_k + 0.1)
plt.savefig('plots/KMEANS.png', dpi = 256)
plt.show()
plt.close()


# ---------------------Affinity Propagation--------------------------
# 
# More propensity -> Points will be more reluctant to becoming prototypes -> Fewer Clusters
# Number of clusters also depends on how the date is distributed
# 
# Reponsibility -> Messages points are sending to each other to indicate how suitable the recepient
# of the message is as a prototype for the sender of the message. It is like a vote -> points vote
# on their representative.
# 
# Availability -> If a point is available to represent the other points

apMetrics = pd.DataFrame(columns = ['damping', 'randIndex', 'precision', 'recall',\
                                        'f1', 'adjustedRandIndex', 'silhouette'])

min_damping = 0.5
max_damping = 1
dampingRange = np.arange(min_damping, max_damping, 0.01)

labels = labels.flatten()
for d in dampingRange:
    
    ap = AffinityPropagation(damping = d, random_state = 5, max_iter = 1000).fit_predict(data_AP)
    
    randIndex = rand_score(labels[labels != 0], ap[labels != 0])
    ci = ClusterIndicators(ap[labels!=0], labels[labels != 0])
    precision = ci.precision
    recall = ci.recall
    f1 = ci.f1

    label_1 = ap[0]
    if(len(data_AP) == len(ap[ap == label_1])):
        #silhouette score throws exception when all points are labeled the same
        #CLEVER trick to get around this and calculate silhouette score
        ap[0] = 1

    silhouette_score_ap = silhouette_score(data_AP, ap)
    
    adjustedRandScore = adjusted_rand_score(labels[labels != 0], ap[labels != 0])

    apMetrics.loc[d] = d, randIndex, precision, recall, f1, adjustedRandScore, silhouette_score_ap


#ver qual o eps cujo valor da silhueta e o mais alto
indexMaxSilhouette = apMetrics["silhouette"].idxmax()
maxSilhouette = apMetrics["silhouette"].max()

print("Affinity Propagation - Max Silhouette: ", maxSilhouette)
print("Affinity Propagation - Max Adjusted Rand Index", apMetrics["adjustedRandIndex"].max())
print("Affinity Propagation - Max Precision", apMetrics["precision"].max())
print("Affinity Propagation - Max Recall", apMetrics["recall"].max())
print("Affinity Propagation - Max F1", apMetrics["f1"].max())
print("Affinity Propagation - Max Rand Index", apMetrics["randIndex"].max())

bestDamping = round(apMetrics["damping"].loc[indexMaxSilhouette], 4)
print("Affinity Propagation - Best Damping: ", bestDamping)

apOtimizado = AffinityPropagation(damping = bestDamping, random_state = 5).fit_predict(data_AP)
tp2_aux_altered.report_clusters(loadLabels[:,0], apOtimizado, 'html/affinityPropagation_bestDamping=' \
                        + str(bestDamping) +'.html')

plt.plot(apMetrics.iloc[:,0],apMetrics.iloc[:,1],label = 'Rand Index')
plt.plot(apMetrics.iloc[:,0],apMetrics.iloc[:,2],label = 'Precision')
plt.plot(apMetrics.iloc[:,0],apMetrics.iloc[:,3],label = 'Recall')
plt.plot(apMetrics.iloc[:,0],apMetrics.iloc[:,4],label = 'F1')
plt.plot(apMetrics.iloc[:,0],apMetrics.iloc[:,5],label = 'Adjusted Rand Index')
plt.plot(apMetrics.iloc[:,0],apMetrics.iloc[:,6],label = 'Silhouette')
plt.scatter(bestDamping, maxSilhouette, color = 'red', label = 'Damping chosen = ' + str(bestDamping))
plt.legend(loc = 'upper left')
plt.xlabel('Damping values')
plt.ylabel('Indexes Values')
plt.title('Indexes Values vs Number of Clusters')
plt.ylim(0, 1.1)
plt.xlim(min_damping, max_damping)
plt.savefig('plots/AFFINITYPROPAGATION.png', dpi = 256)
plt.show()
plt.close()




