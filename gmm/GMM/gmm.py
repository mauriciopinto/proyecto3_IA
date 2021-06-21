import numpy as np
import matplotlib.pyplot as plt
import json
import math

from numpy.core.fromnumeric import transpose

categories = {
    "kidney": 1,
    "hippocampus": 2,
    "cerebellum": 3,
    "colon": 4,
    "liver": 5,
    "endometrium": 6,
    "placenta": 7
}

def parse_class (category):
    return categories[category]

def dot (vector_1, vector_2):
    return sum ([e[0] * e[1] for e in zip(vector_1, vector_2)])

class GMM:

    def __init__(self, n_clusters, n_dims, classed_data):
        self.n_clusters = n_clusters
        self.n_dims = n_dims
        self.classed_data = classed_data


    def p (self, x, mean, covariance):
        rest = [x[i] - mean[i] for i in range (self.n_dims)]
        dotp = dot (rest, rest)
        exp = -dot ([x[i] - mean[i] for i in range (self.n_dims)], [x[i] - mean[i] for i in range (self.n_dims)])
        num = math.e**(exp / (2*(covariance)**2))
        den = covariance * math.sqrt (2*math.pi)
        return num / den


    def gmm_generate_means (self):
        means = [
            [-4.311822, 117.134418],
            [-100.440674, 19.032772],
            [30.423453, -30.253341],
            [-75.992345, 32.658961],
            [95.787545, 69.082386],
            [-5.544384, -105.839754],
            [-75.992345, 32.658961]
        ]
        #means = np.zeros ((self.n_clusters, self.n_dims))
        #for i in range (self.n_clu]sters):
        #    for j in range (self.n_dims):
        #        means[i][j] = np.random.randint (low=-120, high=150)
        return means
        #return k_means (self.n_clusters)


    def gmm_generate_prior (self):
        prior = [
            0.206349,
            0.164021,
            0.201058,
            0.179894,
            0.137566,
            0.079365,
            0.031746
        ]
        return prior


    def gmm_generate_covariance (self, means, x):
        covariance = np.zeros (self.n_clusters)
        for i in range (self.n_clusters):
            variance = sum ([dot ([(self.classed_data[i+1][j][k] - means[i][k]) for k in range (self.n_dims)], [(self.classed_data[i+1][j][k] - means[i][k]) for k in range (self.n_dims)]) for j in range (len (self.classed_data[i+1]))]) / len (self.classed_data[i+1])
            covariance[i] = math.sqrt (variance)
        return covariance


    def gmm_calculate_means (self, gamma, x):
        means = np.zeros ((self.n_clusters, self.n_dims))
        for i in range (self.n_clusters):
            for j in range (self.n_dims):
                num = 0
                for k in range (len (x)):
                    #print ("x")
                    #print (x[k][j])
                    #print ("gamma")
                    #print (gamma[i][k])
                    #print ("product")
                    #print (gamma[i][k] * x[k][j])
                    num += gamma[i][k] * x[k][j]
                    #print ("accumulate")
                    #print (num)
                den = self.gmm_calculate_n (i, gamma, x)
                means[i][j] = num / den
        #print ("------------")
        #print ("------------")
        #print ("------------")
        return means


    def gmm_calculate_prior (self, gamma, x):
        prior = np.zeros (self.n_clusters)
        for i in range (self.n_clusters):
            prior[i] = self.gmm_calculate_n (i, gamma, x) / len (x)
        return prior
        

    def gmm_calculate_covariance (self, gamma, x, mean):
        covariance = np.zeros (self.n_clusters)
        for i in range (self.n_clusters):
            covariance[i] = sum ([gamma[i][j] * dot ([x[j][k] for k in range (self.n_dims)], [x[j][k] for k in range (self.n_dims)]) for j in range (len (x))]) / self.gmm_calculate_n (i, gamma, x) - dot (mean[i], mean[i])
        return covariance


    def gmm_calculate_gamma (self, prior, covariance, x, mean):
        gamma = np.zeros ((self.n_clusters, len (x)))
        for i in range (self.n_clusters):
            for j in range (len (x)):
                p_result = self.p (x[j], mean[i], covariance[i])
                num = prior[i] * self.p (x[j], mean[i], covariance[i])
                den = sum (prior[k] * self.p(x[j], mean[k], covariance[k]) for k in range (self.n_clusters))
                #print (num)
                gamma[i][j] = num / den
        return gamma


    def gmm_calculate_n (self, i, gamma, x):
        n = 0
        for j in range (len (x)):
            n += gamma[i][j]
        return n


    def run_gmm (self, iterations, x, aux=None):
        means = self.gmm_generate_means ()
        prior = self.gmm_generate_prior ()
        covariance = self.gmm_generate_covariance (means, x)
        
        for i in range (iterations):
            #print (sum(prior))
            #print (covariance)
            
            #print (covariance)
            gamma = self.gmm_calculate_gamma (prior, covariance, x, means)
            means = self.gmm_calculate_means (gamma, x)
            prior = self.gmm_calculate_prior (gamma, x)
            covariance = self.gmm_calculate_covariance (gamma, x, means)
            
        #print (prior)
        return means

x_ds = [[float (n) for n in line.split(',')[1:3]] for line in open ('df_pc.csv', 'r').read ().splitlines ()[1:]]
y_ds = [parse_class (line.split(',')[1][1:][:-1]) for line in open ('clase.csv', 'r').read ().splitlines ()[1:]]

prior = np.zeros (7)

count = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: []
}

for y in y_ds:
    count[y].append (x_ds[y]) 


gmm_executer = GMM (7, 2, count)
centroids = gmm_executer.run_gmm (100, x_ds)


def dist (vector_1, vector_2):
    return math.sqrt (sum ((e[1] - e[0])**2 for e in zip (vector_1, vector_2)))

def get_accuracy (vector_1, vector_2):
    hits = 0
    for e in zip (vector_1, vector_2):
        if e[0] == e[1]:
            hits += 1
    return hits / len (vector_1)

#print (centroids)

y_pd = []
for x in x_ds:
    low = 100000000
    ind = 0
    #print ("X")
    #print (x)
    #print ("Centroids")
    #print (centroids)
    for i in range (len (centroids)):

        #print (centroids[i])
        if dist (x, centroids[i]) < low:    
            low = dist (x, centroids[i])
            ind = i
            #print ("lower than current low")
            
    y_pd.append (ind)

#print (y_pd)

print (get_accuracy (y_ds, y_pd))

"""
#print (centroids)
y_pd = []

for i in range (len (x_ds)):
    max = -1
    ind = -1
    for j in range (7):
        if centroids[j][i] > max:
            max = centroids[j][i]
            ind = j
    y_pd.append (ind)

#print (y_pd)
"""