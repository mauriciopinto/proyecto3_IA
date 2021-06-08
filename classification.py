import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from math import sqrt



def dist(el1, el2):
  return sqrt((el2 - el1)**2)

def generatecenters(k):
    cluster_centers={}
    for i in range (k):
        cluster_centers[str(i)] = np.random.randint (0, 255) 
    return cluster_centers

def classify (el, centers,img):
  #print ("new element")
  min = 1000000

  for key in centers:
    if dist(el, centers[key]) < min:
      min = dist(el, centers[key])
      cluster = key
  return cluster



def kmeans(img,k,cluster_centers):    
  classes = {}
  for i in range (k):
      classes[str(i)] = []
  for x in range(img.size[0]):
      for y in range(img.size[1]):
          rgb = img.getpixel ((x, y))
          element = list ([x, y, rgb])
          assigned_cluster = classify (rgb, cluster_centers,img)
          classes[assigned_cluster].append (element)


def dbscan(img,clusters,radius,tree):
  for i in range (len(img)):
    assigned_cluster = -1 
    if clusters[i] == -1:
      assigned_cluster = i 
    else:
      assigned_cluster = clusters[i]
    ind = tree.query_radius([[img[i][0]]], r=radius, count_only=False, return_distance=False)
    is_cluster = False
    if len(ind) >= 4:
      is_cluster = True
    if is_cluster:
      for m in ind:
        clusters[m] = assigned_cluster
    return clusters

def mean(c): 
  return sum ([x[2] for x in c]) / len (c)

def generate_centroids(img):
    centroids = []
    for i in range (3):
        for j in range (3):
            x = i *(int (img.size[0] / 3)) + int ((img.size[0] / 3) / 2)
            y = j *(int (img.size[1] / 3)) + int ((img.size[1] / 3) / 2)
            centroids.append (img.getpixel ((x, y)))
    return centroids

def mean_shift(img,radius,centroids):
    for centroid in centroids:
        for i in range (10):
            neighbors = []
    for x in range (img.size[0]):
      for y in range (img.size[1]):
          if dist (centroid, img.getpixel ((x, y))) <= radius:
              neighbors.append (img.getpixel ((x, y)))
    centroid = mean ()
    