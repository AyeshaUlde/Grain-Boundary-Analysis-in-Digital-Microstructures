import numpy as np
import random
import matplotlib.pyplot as plt
from time import time

n = 256 #grid size
ng = 100 #number of grains
#initialize the digital microstructure as a nxn grid 
arr = np.zeros((n,n), dtype=np.int32)

#create ng nucleation sites in the microstructure by assigning the 
#first ng whole numbers to random pixels
arr[np.random.randint(n, size=ng),np.random.randint(n, size=ng)] = np.arange(1,ng+1)

#function to visualize the grid as heatmap
def heatmap(a,str):
    plt.imshow(a)
    plt.xlim(0,a.shape[0]-1)
    plt.ylim(0,a.shape[1]-1)
    plt.colorbar()
    plt.savefig(str,bbox_inches='tight')
    plt.close()
    #plt.show()

#visualize the microstructure after the generation of ng nucleation sites    
heatmap(arr,'1.png')

#function to find the indices of the nearest non-zero pixel (i.e., nucleation site)
def nearest_nonzero_idx(a,x,y):
    tmp = a[x,y]
    a[x,y] = 0
    r,c = np.nonzero(a)
    a[x,y] = tmp
    min_idx = ((r - x)**2 + (c - y)**2).argmin()
    return r[min_idx], c[min_idx]

#Replacing 0 pixels with the value of the nearest non-zero pixel
arr1 = np.copy(arr)
for i in range(n):
    for j in range(n):
        if(arr[i,j]==0):
            ri, rj = nearest_nonzero_idx(arr,i,j)
            arr1[i,j] = arr[ri,rj]
heatmap(arr1,'2.png')

#function to calculate the grain boundary pixels
def cal_frac_gbp(arr):
    gbp = 0 #grain boundary pixels
    n1 = arr.shape[0]
    n2 = arr.shape[1]
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if(i==0):
                if(j==0):
                    neigh = arr[0:2,0:2] - arr[i,j]
                elif(j==arr.shape[1]-1):
                    neigh = arr[0:2,-2:] - arr[i,j]
                else:
                    neigh = arr[0:2,j-1:j+2] - arr[i,j]
            elif(i==arr.shape[0]-1):
                if(j==0):
                    neigh = arr[-2:,0:2] - arr[i,j]
                elif(j==arr.shape[1]-1):
                    neigh = arr[-2:,-2:] - arr[i,j]
                else:
                    neigh = arr[-2:,j-1:j+2] - arr[i,j]
            elif(j==0):
                neigh = arr[i-1:i+2,0:2] - arr[i,j]
            elif(j==arr.shape[1]-1):
                neigh = arr[i-1:i+2,-2:] - arr[i,j]
            else:
                neigh = arr[i-1:i+2,j-1:j+2] - arr[i,j]
            #For grain interior pixels, the sum of neigh is zero    
            #For grain boundary pixels, the sum of neigh is non-zero
            if(np.sum(neigh)!=0): 
                gbp+=1
    return gbp/(n1*n2)

a = time()
fraction = cal_frac_gbp(arr1)
b = time()
delta = b - a
print("Execution time = {t} ms".format(t=delta*10**3)) #execution time in milliseconds 

print("Fraction of grain boundary pixels = {frac}".format(frac=fraction))