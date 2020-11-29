# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:38:22 2020

@author: Yuxuan Long, Yuqi Wang
"""

import os
import numpy as np
import scipy.linalg as lin

np.random.seed(2020)

def col_normalize(X):
    return X / np.sqrt(np.sum(X ** 2, axis = 0))

def tensor2matrix(X, mode, tensor_dim):
    n = X.shape[tensor_dim - mode]
    X = np.moveaxis(X, tensor_dim - mode, -1)
    return np.reshape(X, (n, -1))

def matrix2tensor(X1, out_shape):
    # out_shape should be like (n3, n2, n1)
    return np.reshape(X1.T, out_shape)

def ALS_solver(X, r, nmax = 1000, err_tol = 1e-4):
    n3, n2, n1 = X.shape
    B = np.random.normal(0, 1, (n2, r))
    C = np.random.normal(0, 1, (n3, r))
    
    X1 = tensor2matrix(X, 1, 3)
    X2 = tensor2matrix(X, 2, 3)
    X3 = tensor2matrix(X, 3, 3)
    
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
    n = A.shape[0]
    
    I = np.eye(n)
    I_big = np.eye(n ** 2)
    A_tilde = np.kron(I_big, A) + np.kron(np.kron(I, A), I) + np.kron(A, I_big)
    
    x = lin.solve(A_tilde, np.reshape(X, (-1, 1)))    
    
    err = lin.norm(A_tilde.dot(x) - np.reshape(X, (-1, 1)))
    return np.reshape(x, X.shape), err

def compute_vec_tensor(U, V, W):
    return np.sum(lin.khatri_rao(lin.khatri_rao(U, V), W), axis = 1)

def low_rank_solver(A, tensors, r, nmax = 1000, err_tol = 1e-4):
    A_hat, B_hat, C_hat = tensors
    
    n = A.shape[0]
    
    U = np.random.normal(0, 1, (n, r))
    V = np.random.normal(0, 1, (n, r))
    W = np.random.normal(0, 1, (n, r))
    I = np.eye(n)
    
    V = col_normalize(V)
    V_V = V.T.dot(V)    
    
    err = np.inf
    approx_left = compute_vec_tensor(W, V, A.dot(U)) + compute_vec_tensor(W, A.dot(V), U) + compute_vec_tensor(A.dot(W), V, U)
    

    i = 0
    while (err >= err_tol) and i < nmax:
        W = col_normalize(W)
        W_W = W.T.dot(W)
        tem = np.reshape(A_hat.dot((C_hat.T.dot(W)) * (B_hat.T.dot(V))).T, (-1, 1))
        u = lin.solve(np.kron(W_W * (V.T.dot(A).dot(V)) + V_V * (W.T.dot(A).dot(W)), I) + np.kron(W_W * V_V, A), tem)
        U = np.reshape(u, (r, n)).T
        
        U = col_normalize(U)
        U_U = U.T.dot(U)
        tem = np.reshape(B_hat.dot((C_hat.T.dot(W)) * (A_hat.T.dot(U))).T, (-1, 1))
        v = lin.solve(np.kron(U_U * (W.T.dot(A).dot(W)) + W_W * (U.T.dot(A).dot(U)), I) + np.kron(W_W * U_U, A), tem)
        V = np.reshape(v, (r, n)).T
        
        V = col_normalize(V)
        V_V = V.T.dot(V)
        tem = np.reshape(C_hat.dot((B_hat.T.dot(V)) * (A_hat.T.dot(U))).T, (-1, 1))
        w = lin.solve(np.kron(V_V * (U.T.dot(A).dot(U)) + U_U * (V.T.dot(A).dot(V)), I) + np.kron(V_V * U_U, A), tem)
        W = np.reshape(w, (r, n)).T
        
        approx_left_new = compute_vec_tensor(W, V, A.dot(U)) + compute_vec_tensor(W, A.dot(V), U) + compute_vec_tensor(A.dot(W), V, U)
        err = lin.norm(approx_left - approx_left_new) / lin.norm(approx_left)
        approx_left = approx_left_new
        i += 1
        print('Relative error at iteration ', i, ': ', err)
    
    print('Finished!')
    return U, V, W, approx_left
    

if __name__ == "__main__":
    
    load_tensor = 0
    n = 200 # 20, 200
    
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
        r1 = 4
        r2 = 15
        
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
        

        A_first, B_first, C_first, X_hat_first = ALS_solver(B1, r = r1)
        np.save('./data/A_first.npy', A_first)
        np.save('./data/B_first.npy', B_first)
        np.save('./data/C_first.npy', C_first)
        
        A_second, B_second, C_second, X_hat_second = ALS_solver(B2, r = r2)
        np.save('./data/A_second.npy', A_second)
        np.save('./data/B_second.npy', B_second)
        np.save('./data/C_second.npy', C_second)
    
    
    A = 2 * np.eye(n) + np.diag(-np.ones((n - 1, )), 1) + np.diag(-np.ones((n - 1, )), -1)
    A *= (n + 1) ** 2
    
    # X, err_direct = direct_solver(B1, A)
    # print(err_direct)
    
    p = A_first.shape[1]
    # r = int(np.ceil(p / 3))
    r = 3
    
    U, V, W, approx_left = low_rank_solver(A, [A_first, B_first, C_first], r)
    final_err = lin.norm(approx_left - B1.flatten())
    print(final_err)
    
    
    # x_approx = compute_vec_tensor(U, V, W)
    # X_approx = np.reshape(x_approx, (n, n, n))
    
    # print(lin.norm(x_approx - X.flatten()))