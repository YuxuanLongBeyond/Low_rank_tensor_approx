# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:38:22 2020

@author: Yuxuan Long
"""

import os
import numpy as np
import scipy.linalg as lin
import scipy.sparse as sp 
from scipy.sparse.linalg import spsolve
import time
import warnings
from scipy.stats import ortho_group

np.random.seed(2020)

def col_normalize(X):
    return X / np.sqrt(np.sum(X ** 2, axis = 0))

def tensor2matrix(X, mode):
    """
    Return mu-mode matricization from a given tensor
    """
    num_dim = len(X.shape)
    n = X.shape[num_dim - mode]
    X = np.moveaxis(X, num_dim - mode, -1)
    return np.reshape(X, (-1, n)).T

def matrix2tensor(X1, out_shape):
    """
    Input: 1-mode matricization
    Output: tensor with size like (n3, n2, n1)
    """
    return np.reshape(X1.T, out_shape)

def ALS_solver(X, r, nmax = 1000, err_tol = 1e-4):
    """

    Parameters
    ----------
    X : tensor like B1
    r : tensor rank
    nmax : maximum number of iterations
        The default is 1000.
    err_tol : tolerance for relative residual error, optional
        The default is 1e-4.

    Returns
    -------
    A : matrix with size n by r
    B : matrix with size n by r
    C : matrix with size n by r
    X_hat : approximated tensor with same shape as X

    """
    n3, n2, n1 = X.shape
    B = np.random.normal(0, 1, (n2, r))
    C = np.random.normal(0, 1, (n3, r))
    
    X1 = tensor2matrix(X, 1)
    X2 = tensor2matrix(X, 2)
    X3 = tensor2matrix(X, 3)
    
    X_norm = lin.norm(X1, 'fro')
    err = np.inf
    
    B = col_normalize(B)
    i = 0
    while (err >= err_tol) and i < nmax:
        C = col_normalize(C)
        tem1 = lin.khatri_rao(C, B)
        A, res, rnk, s = lin.lstsq(tem1, X1.T)
        A = A.T
        
        A = col_normalize(A)
        tem2 = lin.khatri_rao(C, A)
        B, res, rnk, s = lin.lstsq(tem2, X2.T)
        B = B.T
        
        B = col_normalize(B)
        tem3 = lin.khatri_rao(B, A)
        C, res, rnk, s = lin.lstsq(tem3, X3.T)
        C = C.T
        
        X_hat1 = A.dot(lin.khatri_rao(C, B).T)
        err = lin.norm(X_hat1 - X1, 'fro') / X_norm
        i += 1
        print('Relative error at iteration ', i, ': ', err)
    X_hat = matrix2tensor(X_hat1, X.shape)
    print('Finished!')
    return A, B, C, X_hat

def direct_solver(X, A):
    """
    Directly solving the linear system without approximation
    """
    n = A.shape[0]
    
    A = sp.csc_matrix(A)
    
    I = sp.eye(n)
    I_big = sp.eye(n ** 2)
    A_tilde = sp.kron(I_big, A) + sp.kron(sp.kron(I, A), I) + sp.kron(A, I_big)
    
    shaped_X = np.reshape(X, (-1, 1))
    
    x = spsolve(A_tilde, shaped_X)    
    # A_tilde = sp.csc_matrix(A_tilde)
    # err = lin.norm(A_tilde.dot(x) - shaped_X)
    return np.reshape(x, X.shape)

def compute_vec_tensor(U, V, W):
    """
    Return vectorized tensor from CP decomposition 
    """

    # return np.sum(lin.khatri_rao(lin.khatri_rao(U, V), W), axis = 1)
    r = U.shape[1]
    out = 0
    for i in range(r):
        out += lin.khatri_rao(lin.khatri_rao(np.expand_dims(U[:, i], 1), np.expand_dims(V[:, i], 1)), np.expand_dims(W[:, i], 1))
    return np.squeeze(out)
    
def energy_norm(A, U):
    """
    Return tr(U' A U)
    """
    return np.sum(U * (A.dot(U)), axis = 0)

def solve_linear_system(M, b, epsilon = 1e-4):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            x = lin.solve(M, b, assume_a = 'pos')
        except lin.LinAlgWarning:
            print('Badly conditioned matrix found, add damping to solve...')
            # x = lin.solve(M.T.dot(M) + epsilon * np.eye(b.shape[0]), M.T.dot(b), assume_a = 'pos')
            x = lin.solve(M + 0.1 * epsilon * np.eye(b.shape[0]), b, assume_a = 'pos')
        except lin.LinAlgError:
            print('Singular matrix found, add damping...')
            x = lin.solve(M + epsilon * np.eye(b.shape[0]), b, assume_a = 'pos')
        return x

def low_rank_solver(A, tensors, X, p, nmax = 200, err_tol = 1e-3, check_period = 2, ortho_flag = 0):
    """
    

    Parameters
    ----------
    A : SPD matrix with size n by n
    tensors : tensor (like B1) in CP format
    X : tensor (like B1) in original form
    p : tensor rank of approximated solution
    nmax : maximum number of iterations
    err_tol : error tolerance
        The default is 1e-3.
    check_period : period to compute the new error
        The default is 2.
    ortho_flag : flag to allow orthogonalized ALS as initialization
        The default is 0.

    Returns
    -------
    U : matrix with size by n by p
    V : matrix with size by n by p
    W : matrix with size by n by p
    approx_left : evaluated left side of the linear equation, in vectorized form

    Note U, V, W are the CP decomposition of the solution
    
    
    """
    A_hat, B_hat, C_hat = tensors
    
    b = X.flatten()
    b_norm = lin.norm(b)
    
    n = A.shape[0]

    # initialization
    U = ortho_group.rvs(n)
    V = ortho_group.rvs(n)
    W = ortho_group.rvs(n)
    
    U = U[:, :p]
    V = V[:, :p]
    W = W[:, :p]
    
    I = np.eye(n)
    
    scale_base = 0.5
    scale = scale_base
    
    
    if ortho_flag:
        lam, Q = lin.eig(A)
        lam = np.real(lam)
        Q_augment = sp.kron(sp.eye(p), Q)
        Q_augment_T = Q_augment.T
        lam_augment = np.kron(np.ones((p, )), lam)
    # A = sp.csc_matrix(A)
    
    err = np.inf

    print('Begin with tensor rank ', p)

    i = 0
    while (err >= err_tol) and i < nmax:
                
        if ortho_flag:
            V, _ = lin.qr(V, mode='economic')
            W, _ = lin.qr(W, mode='economic')
            tem = (A_hat.dot((C_hat.T.dot(W)) * (B_hat.T.dot(V))).T).flatten()
            alpha = energy_norm(A, V) + energy_norm(A, W)
            u =  Q_augment.dot((Q_augment_T.dot(tem)) / (lam_augment + np.kron(alpha, np.ones((n, )))))
            U = np.reshape(u, (p, n)).T
            
            U, _ = lin.qr(U, mode='economic')
            tem = (B_hat.dot((C_hat.T.dot(W)) * (A_hat.T.dot(U))).T).flatten()
            alpha = energy_norm(A, W) + energy_norm(A, U)
            v =  Q_augment.dot((Q_augment_T.dot(tem)) / (lam_augment + np.kron(alpha, np.ones((n, )))))
            V = np.reshape(v, (p, n)).T
            
            V, _ = lin.qr(V, mode='economic')
            tem = (C_hat.dot((B_hat.T.dot(V)) * (A_hat.T.dot(U))).T).flatten()
            alpha = energy_norm(A, V) + energy_norm(A, U)
            w =  Q_augment.dot((Q_augment_T.dot(tem)) / (lam_augment + np.kron(alpha, np.ones((n, )))))
            W = np.reshape(w, (p, n)).T
            
            if i >= 2:
                ortho_flag = 0
        else:
            V = col_normalize(V)
            V_V = V.T.dot(V)
            W = col_normalize(W)
            W_W = W.T.dot(W)
            tem = (A_hat.dot((C_hat.T.dot(W)) * (B_hat.T.dot(V))).T).flatten()
            M = np.kron(W_W * (V.T.dot(A).dot(V)) + V_V * (W.T.dot(A).dot(W)), I) + np.kron(W_W * V_V, A)
            u = solve_linear_system(M, np.expand_dims(tem, axis = 1))
            U = np.reshape(u, (p, n)).T
    
            
            U = col_normalize(U)
            U_U = U.T.dot(U)
            tem = (B_hat.dot((C_hat.T.dot(W)) * (A_hat.T.dot(U))).T).flatten()
            M = np.kron(U_U * (W.T.dot(A).dot(W)) + W_W * (U.T.dot(A).dot(U)), I) + np.kron(W_W * U_U, A)
            v = solve_linear_system(M, np.expand_dims(tem, axis = 1))
            V = np.reshape(v, (p, n)).T
    
            
            V = col_normalize(V)
            V_V = V.T.dot(V)
            tem = (C_hat.dot((B_hat.T.dot(V)) * (A_hat.T.dot(U))).T).flatten()
            M = np.kron(V_V * (U.T.dot(A).dot(U)) + U_U * (V.T.dot(A).dot(V)), I) + np.kron(V_V * U_U, A)
            w = solve_linear_system(M, np.expand_dims(tem, axis = 1))
            W = np.reshape(w, (p, n)).T

        
        i += 1
        if i % check_period == 0:
            approx_left = compute_vec_tensor(W, V, A.dot(U)) + compute_vec_tensor(W, A.dot(V), U) + compute_vec_tensor(A.dot(W), V, U)
            err_next = lin.norm(approx_left - b) / b_norm
            print('Relative error at iteration ', i, ': ', err_next)
            
            
            if np.abs(err - err_next) < err_tol * scale and p < n and err_next > err_tol:
                
                r_delta = min(n - p, round((err_next / err_tol) ** (1)))
                p += r_delta
                print('Error almost unchanged, increase tensor rank to ', p)
                U = np.concatenate((U, np.random.normal(0, 1, (n, r_delta))), axis = 1)
                V = np.concatenate((V, np.random.normal(0, 1, (n, r_delta))), axis = 1)
                W = np.concatenate((W, np.random.normal(0, 1, (n, r_delta))), axis = 1)
                if scale >= 4e-2:
                    scale *= scale_base
            err = err_next
        else:
            print('Iteration ', i, ' finished')
        
    print('All is finished!')
    return U, V, W, approx_left
    

if __name__ == "__main__":
    n = 200
    p0 = round(n / (max(np.log2(n / 25), 0) + 1))   # initial tensor rank  
    use_direct_solve = 0 # flag to run direct solver
    
    load_tensor = 0  # flag to load stored tensors
    if load_tensor:
        B1 = np.load('./data/B1.npy')
        B2 = np.load('./data/B2.npy')
        
        A_first = np.load('./data/A_first.npy')
        B_first = np.load('./data/B_first.npy')
        C_first = np.load('./data/C_first.npy')
        
        A_second = np.load('./data/A_second.npy')
        B_second = np.load('./data/B_second.npy')
        C_second = np.load('./data/C_second.npy')
    else:
        r1 = 4  # for B1
        r2 = 15 # for B2
        
        if not os.path.exists('./data'):
            os.makedirs('./data')    
        
        B1 = np.zeros((n, n, n))
        B2 = np.zeros((n, n, n))
        zeta = lambda i: i / (n + 1)
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    B1[k, j, i] = np.sin(zeta(i + 1) + zeta(j + 1) + zeta(k + 1))
                    B2[k, j, i] = np.sqrt(zeta(i + 1) ** 2 + zeta(j + 1) ** 2 + zeta(k + 1) ** 2)
        np.save('./data/B1.npy', B1)
        np.save('./data/B2.npy', B2)
        
        t_start = time.perf_counter()
        A_first, B_first, C_first, X_hat_first = ALS_solver(B1, r = r1)
        np.save('./data/A_first.npy', A_first)
        np.save('./data/B_first.npy', B_first)
        np.save('./data/C_first.npy', C_first)
        
        A_second, B_second, C_second, X_hat_second = ALS_solver(B2, r = r2)
        np.save('./data/A_second.npy', A_second)
        np.save('./data/B_second.npy', B_second)
        np.save('./data/C_second.npy', C_second)
        print('Time spent for the CP decomposition: ', time.perf_counter() - t_start, ' seconds')
    
    
    A = 2 * np.eye(n) + np.diag(-np.ones((n - 1, )), 1) + np.diag(-np.ones((n - 1, )), -1)
    A *= (n + 1) ** 2
    
    if use_direct_solve:
        t_start = time.perf_counter()
        X = direct_solver(B1, A)
        print('Time spent for the direct solver: ', time.perf_counter() - t_start, ' seconds')
    
    print('Solving linear system with B1')
    t_start = time.perf_counter()
    U, V, W, approx_left = low_rank_solver(A, [A_first, B_first, C_first], B1, p0)
    print('Time spent for the low-rank solver (B1): ', time.perf_counter() - t_start, ' seconds')
    final_err = lin.norm(approx_left - B1.flatten()) / lin.norm(B1.flatten())
    print('Final relative error: ', final_err)
    
    print('###################################')
    print('Solving linear system with B2')
    t_start = time.perf_counter()
    U, V, W, approx_left = low_rank_solver(A, [A_second, B_second, C_second], B2, p0)
    print('Time spent for the low-rank solver (B2): ', time.perf_counter() - t_start, ' seconds')
    final_err = lin.norm(approx_left - B2.flatten()) / lin.norm(B2.flatten())
    print('Final relative error: ', final_err)   
    