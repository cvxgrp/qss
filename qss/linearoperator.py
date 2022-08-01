import numpy as np
import scipy as sp


class LinearOperator:
    def __init__(self, A):
        self._nrows = 0
        self._ncols = 0

        if (type(A) is not list) or (type(A[0]) is not list):
            raise ValueError(
                "A must be a Scipy sparse CSC matrix or a list of lists of linear operators."
            )

        self._rowdims = [None] * len(A)
        self._coldims = [None] * len(A[0])

        for i, row_block in enumerate(A):
            if type(row_block) is not list:
                raise ValueError(
                    "A must be a Scipy sparse CSC matrix or a list of lists of linear operators."
                )
            if len(row_block) != len(self._coldims):
                raise ValueError("Each row of A must have the same number of elements.")

            row_block_height = row_block[0].shape[0]
            self._nrows += row_block_height
            self._rowdims[i] = row_block_height

            for j, block in enumerate(row_block):
                if block.shape[0] != row_block_height:
                    raise ValueError("Dimension mismatch in A.")
                if self._coldims[j] is None:
                    self._coldims[j] = block.shape[1]
                    self._ncols += block.shape[1]
                else:
                    if self._coldims[j] != block.shape[1]:
                        raise ValueError("Dimension mismatch in A.")

        self._A = A
        self.shape = (self._nrows, self._ncols)

    def matvec(self, v):
        if v.shape[0] != self._ncols:
            raise ValueError("Dimension mismatch.")

        res = np.zeros(self._nrows, dtype=np.cfloat)
        row_index = 0
        for i, row_block in enumerate(self._A):
            col_index = 0
            for j, block in enumerate(row_block):
                if type(block) is sp.sparse.linalg._interface._CustomLinearOperator:
                    res[row_index : row_index + self._rowdims[i]] += block.matvec(
                        v[col_index : col_index + self._coldims[j]]
                    )
                else:
                    res[row_index : row_index + self._rowdims[i]] += (
                        block @ v[col_index : col_index + self._coldims[j]]
                    )
                col_index += self._coldims[j]

            row_index += self._rowdims[i]

        return res

    def rmatvec(self, v):
        if v.shape[0] != self._nrows:
            raise ValueError("Dimension mismatch.")

        res = np.zeros(self._ncols, dtype=np.cfloat)
        col_index = 0
        for i, row_block in enumerate(self._A):
            row_index = 0
            for j, block in enumerate(row_block):
                if type(block) is sp.sparse.linalg._interface._CustomLinearOperator:
                    res[row_index : row_index + self._coldims[j]] += block.rmatvec(
                        v[col_index : col_index + self._rowdims[i]]
                    )
                else:
                    res[row_index : row_index + self._coldims[j]] += (
                        block.T @ v[col_index : col_index + self._rowdims[i]]
                    )
                row_index += self._coldims[j]

            col_index += self._rowdims[i]

        return res
