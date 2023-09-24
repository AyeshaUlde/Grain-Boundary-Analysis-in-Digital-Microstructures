import numpy as np
import random
import matplotlib.pyplot as plt
from time import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 256 #grid size
ng = 100 #number of grains

#function to visualize the grid as heatmap
def heatmap(a,str):
    plt.imshow(a)
    plt.xlim(0,a.shape[0]-1)
    plt.ylim(0,a.shape[1]-1)
    plt.colorbar()
    plt.savefig(str,bbox_inches='tight')
    plt.close()
    #plt.show()

#function to find the indices of the nearest non-zero pixel (i.e., nucleation site)
def nearest_nonzero_idx(a,x,y):
    tmp = a[x,y]
    a[x,y] = 0
    r,c = np.nonzero(a)
    a[x,y] = tmp
    min_idx = ((r - x)**2 + (c - y)**2).argmin()
    return r[min_idx], c[min_idx]

if(rank==0):
    #initialize the digital microstructure as a nxn grid 
    arr = np.zeros((n,n), dtype=np.int32)
    #create ng nucleation sites in the microstructure by assigning the 
    #first ng whole numbers to random pixels
    arr[np.random.randint(n, size=ng),np.random.randint(n, size=ng)] = np.arange(1,ng+1)
    #visualize the microstructure after the generation of ng nucleation sites    
    heatmap(arr,'1.png')
    #Replace 0 pixels with the value of the nearest non-zero pixel
    arr1 = np.copy(arr)
    for i in range(n):
        for j in range(n):
            if(arr[i,j]==0):
                ri, rj = nearest_nonzero_idx(arr,i,j)
                arr1[i,j] = arr[ri,rj]
    heatmap(arr1,'2.png')

#function to calculate the grain boundary pixels
def cal_gbp(arr,rank):
    gbp = 0 #grain boundary pixels
    ranges = np.array([[0,arr.shape[0]-1,0,arr.shape[0]-1], [1,arr.shape[0],0,arr.shape[0]-1], 
        [1,arr.shape[0],1,arr.shape[0]], [1,arr.shape[0],0,arr.shape[0]-1]])
    for i in range(ranges[rank,0],ranges[rank,1]):
        for j in range(ranges[rank,2],ranges[rank,3]):
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
    return gbp

#Divide the grid into 4 sub-grids and send each sub-grid to a processor along with its neighbouring row and column
if(rank==0):
    n1 = int(n/2)
    sub = arr1[0:n1+1,0:n1+1]
    comm.send(arr1[n1-1:,0:n1+1],dest=1,tag=1)
    comm.send(arr1[n1-1:,n1-1:],dest=2,tag=1)
    comm.send(arr1[0:n1+1,n1-1:],dest=3,tag=1)
else:
    sub = comm.recv(source=0,tag=1)

comm.Barrier()
wt = MPI.Wtime()

#For every sub-grid,compute the number of grain boundary points
gbp = cal_gbp(sub,rank)
value = comm.reduce(gbp, op=MPI.SUM, root=0)

wt = MPI.Wtime() - wt

if(rank==0):
    print("Execution time = {t} ms".format(t=wt)) #execution time in milliseconds 
    fraction = value/(np.square(n))
    print("Fraction of grain boundary pixels = {frac}".format(frac=fraction))