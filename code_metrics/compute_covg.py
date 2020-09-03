from sklearn.neighbors import KDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import  numpy as np
from scipy import stats
import time
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

def mean_normalize(x,mu=None, std=None):
    # import pdb;pdb.set_trace()
    if mu is None and std is None:
        mu = np.mean(x)
        std = np.std(x)
    return mu,std, (x-mu)/(std+1e-20)

def metrics(In, Out,KNN = 4, dictionary =None, name=None,path=None):

    In  = torch.load(In)
    out  = torch.load(Out)
    X_out = out ['features']
    X_in = In ['features']

    target_in = np.asarray(In['targets'])
    y_out = np.asarray(out ['estimation'])

    # the features can be extracted from different layers, but we consider the penultimate layer of the CNN
    for layer in range(0,1):
        fea_in = X_in[layer]
        fea_out = X_out[layer]
        fea_in = fea_in.reshape(len(fea_in), -1)

        # to normalize the samples with training data's mean and its std
        mu, std, fea_in = mean_normalize(fea_in, mu=None, std=None)

        fea_out = fea_out.reshape(len(fea_out), -1)
        _,_, fea_out = mean_normalize(fea_out, mu, std)
        print ('in '+str(fea_in.shape)+' out '+str(fea_out.shape))


        idx = np.random.choice(len(fea_out),10000)
        fea_out = fea_out[idx]
        y_out = y_out[idx]
        print ('in '+str(fea_in.shape)+' out '+str(fea_out.shape))


        # for different values for K-NN
        for knn in KNN:
            dictionary.append({'name':name.split('/')[-1]+str(knn)})
            dictionary[-1]['entropy'] = entropy(y_out, target_in)

            A, coverag_distance = adjacency_matrix(fea_in, fea_out, knn)
            dictionary[-1]['cov-distance_all'] = coverag_distance
            dictionary[-1]['cov-ratio_all'] = coverage_number(A)


    return dictionary, (fea_in),(fea_out)



def adjacency_matrix( X_in, X_out, knn):

    nbrs = NearestNeighbors(n_neighbors=knn).fit(X_in)
    dist, indices_in = nbrs.kneighbors(X_out)

    # print('\t\t\t Finding knn  in {:.5f}s'.format(time.time() - t))
    A = np.zeros((len(X_in), len(X_out)))

    for j_out in range(len(indices_in)):
        for i_in in indices_in[j_out]:
            l2 = np.sqrt(np.sum((X_in[i_in,:] - X_out[j_out,:])**2))
            A[i_in, j_out] = l2
    temp = A[A != 0]

    return A, np.sum(temp)/len(temp)



def coverage_number (A):
    count = 0
    count2= 0
    for i in range(A.shape[0]):
        if np.sum(A[i,:])>0: count+=1
        if np.sum(A[i,:])>2: count2+=1

    print (' \t\t\t 1-coverage percentage %f 3-coverage percentage %f'%( count/float(A.shape[0]), count2/float(A.shape[0]) ))
    return count/A.shape[0]

def entropy (y, targets):
    classes =np.unique(targets)
    f_c  = [np.sum(y==cls) for cls in classes]

    p_c = [np.sum(y==cls)/len(y) for  cls in classes]

    print ('Out-dist entropy '+str(stats.entropy(p_c))+' \t'+str(p_c))
    return stats.entropy(p_c)


