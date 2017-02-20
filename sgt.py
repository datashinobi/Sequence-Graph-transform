
# coding: utf-8

# In[14]:

import numpy as np


def getpositions(S, V):
    
    '''
    compute index position of sequence S within V
    
    sequence S , space set V
    
    return list of tuples [(value, position)]
    
            [(209981, (array([8]),))(320033, (array([6]),)]
    '''
    
    positions = [(v, np.where(S==v)) for v in V if v in S]
    
    return positions
    
    
def sgt(S, V, ls, k =1):
    
    '''
    
    Extract Sequence Graph Transform features algorithm 2
    
    
    S: sequence 
    V : set domain of all values
    ls: is length sensitive 
    k: hyperparameter  defaults to 1 for supervised learning typically selected κ from {1, 5, 10}
    
    return: sgt matrix
    
    '''
    
    size  = V.shape[0]
    l = 0
    W0, Wk = np.zeros((size,size)),  np.zeros((size,size))
    positions = getpositions(S,V)
    
    for i, u in enumerate(V):
        try:
            index = [p[0] for p in positions].index(u)
    
        except ValueError:
            # move to next element
            break
        
        U = np.array(positions[index][1]).ravel()
        
        for j, v in enumerate(V):
            
            try:
                index = [p[0] for p in positions].index(v)
            except ValueError:
                # move to next element
                break
            
            V2 = np.array(positions[index][1]).ravel()
        
            C = [(i,j) for i in U for j in V2 if j > i]
            
            W0[i,j] = len(C)
        
            cu = np.array([i[0] for i in C]) 
       
            cv = np.array([i[1] for i in C]) 
       
            Wk[i,j] = np.sum(np.exp(-k * np.abs(cu, cv)))
        
        l += len(U)
    
    if ls:
        W0 /= l
        
    W0[np.where(W0==0)] = 1e7  #avoid divide by 0
    sgt = np.power(np.divide(Wk, W0), 1/k)
    
    
    return sgt

