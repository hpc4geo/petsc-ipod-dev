

import numpy as np


def mgs(A, inplace=False):
    """
    Modified Gram Schmidt.
    """
    nr, nc = A.shape
    if inplace == False:
        Q = np.copy(A)
    else:
        Q = A

    for j in range(0, nc):
        vj = Q[:, j]
        rjj = np.linalg.norm(vj)
        qj = vj / rjj
        for k in range(j+1, nc):
            vk = Q[:, k]
            rjk = np.dot(qj, vk)
            vk += -rjk * qj
        Q[:, j] = qj
    return Q


def qrmgs(A, inplace=False):
    """
    QR Modified Gram Schmidt.
    """
    nr, nc = A.shape
    R = np.zeros((nc, nc))
    if inplace == False:
        #Q = np.zeros(A.shape)
        Q = np.copy(A)
    else:
        Q = A

    for j in range(0, nc):
        vj = Q[:, j]
        rjj = np.linalg.norm(vj)
        qj = vj / rjj
        R[j,j] = rjj
        for k in range(j+1, nc):
            vk = Q[:, k]
            rjk = np.dot(qj, vk)
            R[j,k] = rjk
            vk += -rjk * qj
        Q[:, j] = qj
    return Q, R





A0 = np.array([[1.0,2.0, 3.0, 44.0],[4.0, 55.0, 6.0, 77.0],[14.0, 5.0, 16.0, 97.0]]).T
A = np.array([[1.0,2.0, 3.0, 44.0],[4.0, 55.0, 6.0, 77.0],[14.0, 5.0, 16.0, 97.0]]).T

q,r = np.linalg.qr(A)
print('q',q)
print('r', r)
print(np.max(np.absolute(A - q@r)))

Q,R = qrmgs(A0, inplace=True)
print('Q',Q)
print('R',R)
print(np.max(np.absolute(A - Q@R)))
print('Q^T Q',Q.T @ Q)

A0 = np.array([[1.0,2.0, 3.0, 44.0],[4.0, 55.0, 6.0, 77.0],[14.0, 5.0, 16.0, 97.0]]).T
Q = mgs(A0, inplace=True)
print('Q',Q)
print('Q^T Q',Q.T @ Q)

A = np.array([[1.0,2.0, 3.0, 44.0],[4.0, 55.0, 6.0, 77.0],[14.0, 5.0, 16.0, 97.0]]).T
Q = mgs(A, inplace=False)
print('Q',Q)
err = Q.T @ Q - np.diag(np.ones(A.shape[1]))
print('err',np.max(np.absolute(err)))


def qrmgs2(X, inplace=False):
    """
    QR Modified Gram Schmidt (BLAS 2)
    """
    nr, nc = X.shape
    R = np.zeros((nc, nc))
    T = np.zeros((nc, nc))
    if inplace == False:
        Q = np.copy(X)
    else:
        Q = X

    xk = X[:, 0]
    rkk = np.linalg.norm(xk)
    qk = np.copy(xk) / rkk
    R[0, 0] = rkk
    Q[:, 0] = qk[:]
    T[0, 0] = 1.0

    Qhat = Q[:, 0]
    Qhat = np.atleast_2d(Qhat).T

    That = T[0:1, 0:1]
    That = np.atleast_2d(That)

    Rhat = R[0:1, 0:1]
    Rhat = np.atleast_2d(Rhat)
    print('Q', Qhat.shape, Qhat.flags['OWNDATA'], 'T', That.shape, That.flags['OWNDATA'], 'T', Rhat.shape, Rhat.flags['OWNDATA'])

    for k in range(1, nc):
        xk = Q[:, k] # This is really X
        hk = np.matmul(Qhat.T, xk)
        hk = np.atleast_1d(hk) # Needed at first iteration as hk will be a scalar rather than an ndarray
        hk = That.T @ hk
        hk = np.atleast_1d(hk)
        #print(hk.shape,Qhat.shape)
        print('  hk', hk.shape)

        yk = xk - Qhat @ hk

        rkk = np.linalg.norm(yk)
        qk = np.copy(yk) / rkk

        gk = np.matmul(Qhat.T, qk)
        gk = np.atleast_1d(gk) # Needed at first iteration as hk will be a scalar rather than an ndarray
        gk = -That @ gk
        gk = np.atleast_1d(gk)
        print('  gk', gk.shape)

        Q[:, k] = qk[:]
        Qhat = Q[:, 0:k+1] # update view

        nvals = gk.shape[0]
        T[0:nvals, k] = gk[:]
        T[k, k] = 1.0
        That = T[0:k+1, 0:k+1] # update view
        print('  That', That)

        nvals = hk.shape[0]
        R[0:nvals, k] = hk[:]
        R[k, k] = rkk
        Rhat = R[0:k+1, 0:k+1] # update view
        print('  Rhat', Rhat)

        print('ITER', 'Q', Qhat.shape, Qhat.flags['OWNDATA'], 'T', That.shape, That.flags['OWNDATA'], 'T', Rhat.shape, Rhat.flags['OWNDATA'])

    return Q, R, T

A = np.array([[1.0,2.0, 3.0, 44.0],[4.0, 55.0, 6.0, 77.0],[14.0, 5.0, 16.0, 97.0]]).T
Q, R, _ = qrmgs2(A, inplace=False)
A = np.array([[1.0,2.0, 3.0, 44.0],[4.0, 55.0, 6.0, 77.0],[14.0, 5.0, 16.0, 97.0]]).T
print('Q',Q)
err = Q.T @ Q - np.diag(np.ones(A.shape[1]))
print('err',np.max(np.absolute(err)))
print(np.max(np.absolute(A - Q@R)))
