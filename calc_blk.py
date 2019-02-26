"""
The MIT License (MIT)
Copyright (c) 2015 Evan Archer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


# import theano
import numpy as np
# import theano.tensor as T
# # from theano.tensor.shared_randomstreams import RandomStreams
# import theano.tensor.nlinalg as Tla
# import theano.tensor.slinalg as Tsla
import torch as tc


def blk_tridag_chol(A, B):
    """
    Compute the cholesky decompoisition of a symmetric, positive definite
    block-tridiagonal matrix.

    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper) 1st block
        off-diagonal matrix

    Outputs:
    R - python list with two elements
        * R[0] - [T x n x n] tensor of block diagonal elements of Cholesky decomposition
        * R[1] - [T-1 x n x n] tensor of (lower) 1st block off-diagonal elements of Cholesky

    """
    T = A.shape[0]
    n = A.shape[1]
    assert(A.shape == (T, n, n))
    assert(B.shape == (T-1, n, n))

    # Code for computing the cholesky decomposition of a symmetric block tridiagonal matrix
    def compute_chol(Aip1, Bi, Li):
        Ci = Bi @ tc.inverse(Li).t()
        # Ci = T.dot(Bi.T, Tla.matrix_inverse(Li).T)
        Dii = Aip1 - Ci @ Ci.t()
        # Dii = Aip1 - T.dot(Ci, Ci.T)

        Lii = tc.cholesky(Dii)
        # Lii = Tsla.cholesky(Dii)
        return [Lii, Ci]

    L = tc.empty_like(A)
    C = tc.empty_like(B)
    L[0] = tc.cholesky(A[0])
    # L1 = Tsla.cholesky(A[0])
    C[0] = tc.zeros_like(B[0])
    # C1 = T.zeros_like(B[0])
    for t in range(0, len(B)):  # T-1
        L[t+1], C[t] = compute_chol(A[t+1], B[t], L[t])
    # this scan returns the diagonal and off-diagonal blocks of the cholesky decomposition
    # mat, updates = theano.scan(fn=compute_chol, sequences=[A[1:], B], outputs_info=[L1,C1])
    return [L, C]


def blk_chol_inv(A, B, b, lower=True, transpose=False):
    """
    Solve the equation Cx = b for x, where C is assumed to be a
    block-bi-diagonal matrix ( where only the first (lower or upper)
    off-diagonal block is nonzero.

    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper or lower)
        1st block off-diagonal matrix

    lower (default: True) - boolean specifying whether to treat B as the lower
          or upper 1st block off-diagonal of matrix C
    transpose (default: False) - boolean specifying whether to transpose the
          off-diagonal blocks B[i,:,:] (useful if you want to compute solve
          the problem C^T x = b with a representation of C.)

    Outputs:
    x - solution of Cx = b
    """
    T = A.shape[0]
    n = A.shape[1]
    # assert(A.shape == (T, n, n))
    # assert(B.shape == (T-1, n, n))
    # print(b.shape)
    # assert(b.shape == (T, n,1))

    if transpose:
        A = tc.einsum('tjk->tkj', A)
        B = tc.einsum('tjk->tkj', B)
    if lower:
        x = tc.zeros((T, n, n))
        bt = tc.zeros((T, n, n))
        # b0 = b[0]
        # a0 = tc.inverse(A[0])
        # print(a0)
        # print(b0)
        # print(tc.inverse(a0) @ b0)
        x[0] = tc.inverse(A[0]) @ b[0]

        def lower_step(Akp1, Bk, bkp1, xk):
            return tc.inverse(Akp1) @ bkp1 - Bk @ xk

        for t in range(T-1):
            x[t+1] = lower_step(A[t+1], B[t], b[t+1], x[t].clone())  # clone prevents inplace error
            # x[t+1] = x[t+1] + tc.inverse(A[t+1]) @ b[t+1]
            # bt[t] = tc.einsum('ij,jk->ik',B[t].clone(),x[t].clone())
            # x[t+1] = x[t+1] - bt[t]
    else:
        x = tc.zeros((T, n))
        x[-1] = tc.inverse(A[-1]) @ b[-1]

        def upper_step(Akm1, Bkm1, bkm1, xk):
            return tc.inverse(Akm1) @ (bkm1 - Bkm1 @ xk)

        for t in range(T-1):
            x[-t-2] = upper_step(A[-t-2], B[-t-1], b[-t-2], x[-t-1].clone())
    return x


def blk_chol_mtimes(A, B, x, lower=True, transpose=False):
    """
    Evaluate Cx = b, where C is assumed to be a
    block-bi-diagonal matrix ( where only the first (lower or upper)
    off-diagonal block is nonzero.

    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper or lower)
        1st block off-diagonal matrix

    lower (default: True) - boolean specifying whether to treat B as the lower
          or upper 1st block off-diagonal of matrix C
    transpose (default: False) - boolean specifying whether to transpose the
          off-diagonal blocks B[i,:,:] (useful if you want to compute solve
          the problem C^T x = b with a representation of C.)

    Outputs:
    b - result of Cx = b

    """
    print('This function is not tested')
    raise NotImplementedError
    T = A.shape[0]
    n = A.shape[1]
    assert(A.shape == (T, n, n))
    assert(B.shape == (T-1, n, n))
    assert(x.shape == (T, n))
    if transpose:
        A = A.dimshuffle(0, 2, 1)
        B = B.dimshuffle(0, 2, 1)
    if lower:
        b = tc.empty((T, n))
        b[0] = A[0] @ x[0]
        # b0 = (A[0]).dot(x[0])

        def lower_step(Ak, Bkm1, xkm1, xk):
            return Bkm1 @ xkm1 + Ak @ xk
            # return Bkm1.dot(xkm1) + Ak.dot(xk)

        for t in range(T-1):
            b[t+1] = lower_step(A[t+1], B[t], x[t], x[t+1])
        # X = theano.scan(fn=lower_step, sequences=[A[1:], B, dict(input=x, taps=[-1, 0])])[0]
        # X = T.concatenate([T.shape_padleft(b0), X])
    else:
        raise NotImplementedError
        # b = tc.empty((T, n))
        # b[T-1] = A[-1] @ x[-1]
        # # bN = (A[-1]).dot(x[-1])
        #
        # def upper_step(Ak, Bk, xkm1, xk):
            # return Ak @ xkm1 + Bk @ xk
            # # return Ak.dot(xkm1) + Bk.dot(xk)
        #
        # for t in range(T-1):
            # b[t+1] = upper_step(A[t+1], B[t], x[t], x[t+1])
        # # X = theano.scan(fn = lower_step, sequences=[A, B, dict(input=x, taps=[-1, 0])])[0]
        # X = T.concatenate([X, T.shape_padleft(bN)])
    return b


if __name__ == "__main__":
    print('oh yeah....')

    # Build a block tridiagonal matrix
    npA = np.mat('1  .9; .9 4', dtype=float)
    npB = .01*np.mat('2  7; 7 4', dtype=float)
    npC = np.mat('3  0; 0 1', dtype=float)
    npD = .01*np.mat('7  2; 9 3', dtype=float)
    npE = .01*np.mat('2  0; 4 3', dtype=float)
    npF = .01*np.mat('1  0; 2 7', dtype=float)
    npG = .01*np.mat('3  0; 8 1', dtype=float)

    npZ = np.mat('0 0; 0 0')

    lowermat = np.bmat([[npF,     npZ, npZ,   npZ],
                        [npB.T,   npC, npZ,   npZ],
                        [npZ,   npD.T, npE,   npZ],
                        [npZ,     npZ, npB.T, npG]])
    print(lowermat)
    # tlower = tc.tensor(lowermat)

    # make lists of theano tensors for the diagonal and off-diagonal blocks
    tA = tc.tensor(npA)
    tB = tc.tensor(npB)
    tC = tc.tensor(npC)
    tD = tc.tensor(npD)
    tE = tc.tensor(npE)
    tF = tc.tensor(npF)
    tG = tc.tensor(npG)
    # tC = theano.shared(value=npC)
    # tD = theano.shared(value=npD)
    # tE = theano.shared(value=npE)
    # tF = theano.shared(value=npF)
    # tG = theano.shared(value=npG)

    theD = tc.stack(tensors=(tF, tC, tE, tG), dim=0)
    theOD = tc.stack(tensors=(tB.t(), tD.t(), tB.t()), dim=0)

    npb = np.mat('1 2; 3 4; 5 6; 7 8', dtype=float)
    print(npb)
    # tb = T.matrix('b')
    tb = tc.tensor(npb)

    cholmat = lowermat.dot(lowermat.T)
    print(cholmat.shape)
    # cholmat = lower @ tlower.t()

    # invert matrix using Cholesky decomposition
    # intermediary -
    # print(tb.shape)
    print('tb {}'.format(tb.shape))
    ib = blk_chol_inv(theD, theOD, tb)

    npb2 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    print(np.linalg.inv(lowermat).dot(npb2))
    # print(ib)
    # print('ib {}'.format(ib.shape))
    # print(ib.shape)
    # final result -
    x = blk_chol_inv(theD, theOD, ib, lower=False, transpose=True)
    print(np.linalg.inv(lowermat.T).dot(ib.flatten()))
    # print(x.shape)
    print('x shape: {}'.format(x.shape))
    print('x {}'.format(x.flatten()))

    # print(np.allclose(ib, np_inv.dot(npb)))

    # x2 = np.linalg.inv(cholmat)  # .dot(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    # print(x2.shape)
    # print(x2.dot(np.array([1, 2, 3, 4, 5, 6, 7, 8])))
    # f = theano.function([tb], x)

    # print('Cholesky inverse matches numpy inverse: ', np.allclose(f(npb).flatten(), np.linalg.inv(cholmat).dot(np.array([1, 2, 3, 4, 5, 6, 7, 8]))))
    print('Cholesky inverse matches numpy inverse: ', np.allclose(x.flatten(), np.linalg.inv(cholmat).dot(np.array([1, 2, 3, 4, 5, 6, 7, 8]))))
