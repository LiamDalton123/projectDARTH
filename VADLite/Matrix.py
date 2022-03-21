import logging
from io import *
import numpy


# Jama = Java Matrix class.
# <P>
#    The Java Matrix Class provides the fundamental operations of numerical
#    linear algebra.  Various constructors create Matrices from two-dimensional
#    arrays of double precision floating point numbers.  Various "gets" and
#    "sets" provide access to submatrices and matrix elements.  Several methods
#    implement basic matrix arithmetic, including matrix addition and
#    multiplication, matrix norms, and element-by-element array operations.
#    Methods for reading and printing matrices are also included.  All the
#    operations in this version of the Matrix Class involve real matrices.
#    Complex matrices may be handled in a future version.
# <P>
#    Five fundamental matrix decompositions, which consist of pairs or triples
#    of matrices, permutation vectors, and the like, produce results in five
#    decomposition classes.  These decompositions are accessed by the Matrix
#    class to compute solutions of simultaneous linear equations, determinants,
#    inverses and other matrix functions.  The five decompositions are:
# <P><UL>
#    <LI>Cholesky Decomposition of symmetric, positive definite matrices.
#    <LI>LU Decomposition of rectangular matrices.
#    <LI>QR Decomposition of rectangular matrices.
#    <LI>Singular Value Decomposition of rectangular matrices.
#    <LI>Eigenvalue Decomposition of both symmetric and non-symmetric square matrices.
# </UL>
# <DL>
# <DT><B>Example of use:</B></DT>
# <P>
# <DD>Solve a linear system A x = b and compute the residual norm, ||b - A x||.
# <P><PRE>
#       double[][] vals = {{1.,2.,3},{4.,5.,6.},{7.,8.,10.}};
#       Matrix A = new Matrix(vals);
#       Matrix b = Matrix.random(3,1);
#       Matrix x = A.solve(b);
#       Matrix r = A.times(x).minus(b);
#       double rnorm = r.normInf();
# </PRE></DD>
# </DL>
#
# @author The MathWorks, Inc. and NIST
# @version 5 August 1998

# This class has been translated to Python for this project.

class Matrix:
    # ------------------------
    #    Constructors
    # ------------------------
    #
    #    Construct an m-by-n matrix of zeros.
    #    @param m    Number of rows.
    #    @param n    Number of columns.

    def __init__(self, m, n):
        # ------------------------
        #    Instance variables
        # ------------------------
        #
        # Row and column dimensions.
        #    @serial row dimension.
        #    @serial column dimension.
        self.m = m
        self.n = n
        #    Array for internal storage of elements.
        #    @serial internal array storage.
        self.A = []
        for i in range(0, self.m):
            lst = []
            for j in range(0, self.n):
                lst.append(0.0)
            self.A.append(lst)

    def getArray(self):
        return self.A

    # Get a submatrix.
    #    @param i0   Initial row index
    #    @param i1   Final row index
    #    @param j0   Initial column index
    #    @param j1   Final column index
    #    @return     A(i0:i1,j0:j1)
    #    @exception ArrayIndexOutOfBoundsException Submatrix indices

    def getMatrix(self, i0, i1, j0, j1):
        X = Matrix(i1 - i0 + 1, j1 - j0 + 1)
        B = X.getArray()
        try:
            for i in range(i0, i1+1):
                for j in range(j0, j1+1):
                    B[i-i0][j-j0] = self.A[i][j]
        except IndexError as err:
            logging.error("Index error: {0}".format(err))
        return X

    # Linear algebraic matrix multiplication, A * B
    # @param B    another matrix
    # @return     Matrix product, A * B
    # @exception IllegalArgumentException Matrix inner dimensions must agree.
    def times (self, B):
        if B.m != self.n:
            raise ValueError("Matrix inner dimensions must agree.")

        X = Matrix(self.m,B.n)
        C = X.getArray()
        Bcolj = [0.0] * self.n
        for j in range (0, B.n):
            for k in range (0, self.n):
                Bcolj[k] = B.A[k][j]

            for i in range(0, self.m):
                Arowi = self.A[i]
                s = 0.0
                for k in range(0, self.n):
                    s += Arowi[k]*Bcolj[k]
                C[i][j] = s
        return X



