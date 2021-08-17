#This package contains several functions based on the boxplot method to eliminate outliers in unidimensional data

import numpy as np

def logbox(x, mult= 1.5):
    xpos = x[x > 0.0]
    if len(xpos) < 3:
        return (-np.inf, np.inf)
    q1q3 = np.quantile(xpos, [0.25, 0.75])
    q1 = np.log(q1q3[0])
    q3 = np.log(q1q3[1])
    iqr = q3 - q1
    lower = np.exp(q1 - mult * iqr)
    upper = np.exp(q3 + mult * iqr)
    return (lower, upper)

from numba import jit
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer

def coxbox(x, mult= 1.5):
    q1q3 = np.quantile(x[x > 0.0], [0.25, 0.75])
    bc = PowerTransformer(method = 'box-cox', standardize= False)
    bc.fit(np.array(x[x > 0.0]).reshape(-1,1))
    q1 = bc.transform(q1q3[0].reshape(1,-1))
    q3 = bc.transform(q1q3[1].reshape(1,-1))
    iqr = q3 - q1
    lower = bc.inverse_transform((q1 - mult * iqr).reshape(1,-1)).squeeze()
    upper = bc.inverse_transform((q3 + mult * iqr).reshape(1,-1)).squeeze()
    return (lower, upper)

@jit('float64[:](int32[:], int32[:], float32[:], float32[:], int32)', nopython= True)
def h_vec(i, j, Zplus, Zminus, p):
    a = Zplus[i].astype(np.float64)
    b = Zminus[j].astype(np.float64)
    
    return np.where(np.abs(a - b) < 5e-16, np.sign(p - 1.0 - i - j), (a + b)/(a - b))

@jit('float64(int32, int32, float32[:], float32[:], int32)', nopython= True)
def h_scalar(i, j, Zplus, Zminus, p):
    a = np.float64(Zplus[i])
    b = np.float64(Zminus[j])

    if np.abs(a - b) < 5e-16:
        return np.sign(p - 1.0 - i - j)
    else:
        return (a + b)/(a - b)
    
@jit('int32[:](int32, int32, float64, float32[:], float32[:])', nopython= True)
def greater_h(p, q, u, Zplus, Zminus):  # int p, int q, real u
    # h is the kernel function, h(i,j) gives the ith, jth entry of H
    # p and q are the number of rows and columns of the kernel matrix H

    # vector of size p
    P = np.empty(p, dtype= np.int32)

    # indexing from zero
    j = 0

    # starting from the bottom, compute the least upper bound for each row
    for i in range(0, p):

        # search this row until we find a value less than u
        while j < q and h_scalar(p-1-i, j, Zplus, Zminus, p) > u + 5e-16:
            j += 1

        # the entry preceding the one we just found is greater than u
        P[p-1-i] = j - 1

    return P

@jit('int32[:](int32, int32, float64, float32[:], float32[:])', nopython= True)
def less_h(p, q, u, Zplus, Zminus):  # function h, int p, int q, real u
    # vector of size p
    Q = np.empty(p, dtype= np.int32)

    # last possible row index
    j = q - 1

    # starting from the top, compute the greatest lower bound for each row
    for i in range(0, p):

        # search this row until we find a value greater than u
        while j >= 0 and h_scalar(i, j, Zplus, Zminus, p) < u - 5e-16:
            j -= 1

        # the entry following the one we just found is less than u
        Q[i] = j + 1

    return Q

def wmedian(values, weights): #Weighted median
    idx_sorted = np.argsort(values[::-1])
    w_sorted = weights[idx_sorted]
    obj = np.sum(w_sorted)/2.0
    val_sorted = values[idx_sorted]

    cumsum = np.cumsum(w_sorted)
    ind = np.argmax(cumsum >= obj)
    
    if w_sorted[-1] > obj:
        ind = -1
    r = val_sorted[ind]
    if abs(cumsum[ind] - obj) < 5e-16:
        r = (val_sorted[ind] + val_sorted[ind+1])/2.0
    return r

def medcouple(X, maxIter = int(1e6)):
    # X is a vector of size n
    Xc = np.asarray(X, dtype= np.float64)

    # compute initial ingredients as for the naïve medcouple
    n = Xc.shape[0]
    if n % 2 == 0:
        ind_m = np.array([n/2 - 1, n/2], dtype= np.int64)
    else:
        ind_m = np.array([(n-1)/2], dtype= np.int64)

    y = np.sort(Xc)[::-1]
    del Xc

    xm = np.mean(y[np.min(ind_m) : np.max(ind_m) + 1])
    maxabs = max(abs(y[0]), abs(y[-1]))
    xscale = 2.0 * maxabs if maxabs > 1e-15 else 1.0

    Zplus = ((y[: np.min(ind_m) + 1] - xm)/xscale).astype(np.float32)
    Zminus = ((y[np.max(ind_m) :] - xm)/xscale).astype(np.float32)

    p = Zplus.shape[0]
    q = Zminus.shape[0]
    
    # begin Kth pair algorithm (Johnson & Mizoguchi)

    # the initial left and right boundaries, two vectors of size p
    L = np.zeros(p, dtype= np.int32)
    R = np.full(p, q-1, dtype= np.int32)

    # number of entries to the left of the left boundary
    Ltotal = 0

    # number of entries to the left of the right boundary
    Rtotal = p * q

    # since we are indexing from zero, the medcouple index is one less than its rank
    medcouple_index = np.int64(np.floor(Rtotal / 2))
    
    aux_P = np.arange(p, dtype= np.int32)   
    # iterate while the number of entries between the boundaries is greater than the number of rows in the matrix    
    it_num = 0
    found = False
    while Rtotal - Ltotal > p and it_num < maxIter:
        it_num += 1
        # compute row medians and their associated weights, but skip any rows that are already empty
        middle_idx = aux_P[L <= R]
        aux_floor = np.floor((L + R)/2.0).astype(np.int32)

        row_medians = h_vec(middle_idx, aux_floor[middle_idx], Zplus, Zminus, p)
        weights = R[middle_idx].astype(np.float64) - L[middle_idx].astype(np.float64) + 1.0
        
        WM = wmedian(row_medians, weights= weights)
                
        # new tentative right and left boundaries
        P = greater_h(p, q, WM, Zplus, Zminus)
        Q = less_h(p, q, WM, Zplus, Zminus)

        Ptotal = np.sum(P.astype(np.int64)) + P.shape[0]
        Qtotal = np.sum(Q.astype(np.int64))
        
        # determine which entries to discard, or if we've found the medcouple
        if medcouple_index <= Ptotal - 1:
            R = P
            Rtotal = Ptotal
        elif medcouple_index > Qtotal - 1:
            L = Q
            Ltotal = Qtotal
        else:
            found = True
            break # found the medcouple, rank of the weighted median equals medcouple index

    # did not find the medcouple, but there are very few tentative entries remaining
    if not found:
        middle_idx = aux_P[L <= R]
        remaining = np.array([h_scalar(i, j, Zplus, Zminus, p) for i in middle_idx for j in range(L[i], R[i] + 1)])

        # select the medcouple by rank amongst the remaining entries
        WM = np.partition(remaining, medcouple_index - Ltotal)[medcouple_index - Ltotal]
    
    return WM

def adjbox(x, mult= 1.5, long_tail = 3.0, short_tail= 4.0):
    if len(x) < 3:
        return (-np.inf, np.inf)
    q1q3 = np.quantile(x, [0.25, 0.75])
    q1 = q1q3[0]
    q3 = q1q3[1]
    iqr = q3 - q1
    mc = medcouple(x)
    if mc >= 0.0:
        rt = long_tail
        lt = short_tail
    else:
        rt = short_tail
        lt = long_tail
    lower = q1 - mult * np.exp(-lt * mc) * iqr
    upper = q3 + mult * np.exp(rt * mc) * iqr
    return (lower, upper)

def logadjbox(x, mult= 1.5, long_tail = 3.0, short_tail= 3.0):
    q1q3 = np.quantile(x[x > 0.0], [0.25, 0.75])
    q1 = np.log(q1q3[0])
    q3 = np.log(q1q3[1])
    iqr = q3 - q1
    mc = medcouple(np.log(x[x > 0.0]))
    if mc >= 0.0:
        rt = long_tail
        lt = short_tail
    else:
        rt = short_tail
        lt = long_tail
    lower = np.exp(q1 - mult * np.exp(-lt * mc) * iqr)
    upper = np.exp(q3 + mult * np.exp(rt * mc) * iqr)
    return (lower, upper)

def yjbox(x, mult= 1.5, long_tail = 3.0, short_tail= 3.0):
    q1q3 = np.quantile(x, [0.25, 0.75])
    yj = PowerTransformer(method = 'yeo-johnson', standardize= False)
    yj.fit(np.array(x).reshape(-1,1))
    q1 = yj.transform(q1q3[0].reshape(1,-1))
    q3 = yj.transform(q1q3[1].reshape(1,-1))
    iqr = q3 - q1
    mc = medcouple(yj.transform(np.array(x).reshape(-1,1)).squeeze())
    if mc >= 0.0:
        rt = long_tail
        lt = short_tail
    else:
        rt = short_tail
        lt = long_tail
    lower = yj.inverse_transform((q1 - mult * np.exp(-lt * mc) * iqr).reshape(1,-1)).squeeze()
    upper = yj.inverse_transform((q3 + mult * np.exp(rt * mc) * iqr).reshape(1,-1)).squeeze()
    return (lower, upper)