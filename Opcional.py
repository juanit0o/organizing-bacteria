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
####################################################################################################

                                    ############## ISOMAP ############

isomap = Isomap(n_components=6)
fitTransformImagesISO = isomap.fit_transform(imagesMatrix)
scaledTransformedDataISO = scaler.fit_transform(fitTransformImagesISO)     

isomap_array = np.append(scaledTransformedDataISO, labels, axis=1)


############################### VISUALIZACAO #####################

df_iso = pd.DataFrame(isomap_array,columns = ['iso_1','iso_2','iso_3','iso_4','iso_5','iso_6','labels'])

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

labels = labels.flatten()


highestIndexesF_KMEANS = sfs(KMeans(n_clusters = 4, random_state = 5), highestIndexesF.copy(), labels)
print("Selected features for KMEANS:", highestIndexesF_KMEANS)



#fazer a ordenacao dos vizinhos
#ir buscar a distancia do quinto vizinho mais perto
vectorOnes = np.ones(len(df.iloc[:,highestIndexesF]))
data_KMEANS = df.iloc[:,highestIndexesF_KMEANS]


##############################
def get_bissecting_kmeans_iteration_result(labels, iteration):
    
    result = []
    
    for l in labels:
        if len(l) > iteration:
            result.append(l[0:iteration])
        else:
            result.append(l)
            
    result = np.array(result, dtype=object)
    return result
    

def plot_bissecting_kmeans_2d(data_0, data_1, scale, name = 'test', folder = 'plots/bissecting_kmeans/', ext = '.png'):
    data_0_np = data_0.to_numpy()
    x_0 = data_0_np[:,0]
    y_0 = data_0_np[:,1]

    plt.plot(x_0, y_0, ".r")
    
    if not data_1.empty:
        data_1_np = data_1.to_numpy()
        x_1 = data_1_np[:,0]
        y_1 = data_1_np[:,1]
        plt.plot(x_1, y_1, ".b")
    
    max_y, min_y, max_x, min_x = scale
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(min_y - 0.5, max_y + 0.5)
    plt.xlim(min_x - 0.5, max_x + 0.5)    
    
    plt.savefig(folder + name + ext, dpi = 256) if name != '' and ext != '' else 0
    plt.show()
    plt.close()
        

def bissecting_kmeans(data, plot=False):
    index_map = dict()
    for i in range(0, len(data)):
        index_map[i] = i
    
    bissecting_kmeans_results = []
    for i in range(0, len(data)):
        bissecting_kmeans_results.append([])
    
    scale = None
    if plot:
        data_np = data.to_numpy()
        x = data_np[:,0]
        y = data_np[:,1]
        
        max_y = max(y)
        min_y = min(y)
        max_x = max(x)
        min_x = min(x)
        
        scale = max_y, min_y, max_x, min_x
    
        
    bissecting_kmeans_aux(data, bissecting_kmeans_results, index_map, '', plot, scale)
    
    # plot_bissecting_kmeans_2d(data_0, data_1) if plot else 0
    
    return bissecting_kmeans_results


def bissecting_kmeans_aux(data, result_matrix, index_map, iteration, plot=False, scale=None):
        
    if len(data) == 1:
        return
    
    result = KMeans(n_clusters = 2, random_state = 5).fit_predict(data)
    result_0 = np.where(result == 0)[0] #indexes of points labeled 0
    result_1 = np.where(result == 1)[0] #indexes of points labeled 0
    
    # result_0 tem sempre o cluster com mais elementos
    if(len(result_1) > len(result_0)):
        aux = result_1.copy()
        result_1 = result_0
        result_0 = aux
    
    
    index_map_0 = dict()
    counter = 0

    for i in result_0:
        result_matrix[ index_map[i] ].append(0)
        index_map_0[counter] = index_map[i]
        counter += 1
        
    data_0 = data.iloc[np.array(result_0),:]
    
    
    index_map_1 = dict()
    counter = 0
    for i in result_1:
        result_matrix[ index_map[i] ].append(1)
        index_map_1[counter] = index_map[i]
        counter += 1
        
    data_1 = data.iloc[np.array(result_1),:]
    

    plot_bissecting_kmeans_2d(data_0, data_1, scale, 'test_' + str(iteration)) if plot else 0
    

    bissecting_kmeans_aux(data_0, result_matrix, index_map_0, iteration + 'A', plot, scale)
    return bissecting_kmeans_aux(data_1, result_matrix, index_map_1, iteration + 'B', plot, scale)

    


result = bissecting_kmeans(data_KMEANS, False)

# for r in result:
#     print(r, "\n")
    
tp2_aux_altered.report_clusters_hierarchical(loadLabels[:,0], result, 'html/bissecting_kmeans.html')

number_of_iterations = len(max(result, key = lambda x: len(x)))

for it in range(1, number_of_iterations + 1):
    
    result_iter = get_bissecting_kmeans_iteration_result(result, it)
    tp2_aux_altered.report_clusters_hierarchical(loadLabels[:,0], result_iter, 'html/bissecting_kmeans/bk_iteration_'\
                                          + str(it) + '.html', 2)




# test = [[1, 2], [2, 2.5], [1.5, 0.5], [0.2, 2], [1, 1.8], [0.2, 1], [0.3, 2], [2.5, 0.5], [0.2, 2.2], [1.9, 1]]
# test = pd.DataFrame(test, columns = ['x','y'])
# result_test = bissecting_kmeans(test, True)

# for r in result_test:
#     print(r, "\n")
    

# result_iter = get_bissecting_kmeans_iteration_result(result_test, 3)
# tp2_aux_altered.report_clusters_hierarchical([0,1,2,3,4,5,6,7,8,9], result_iter, 'html/test.html')


