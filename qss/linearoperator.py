import numpy as np
import scipy as sp


class LinearOperator:
    # Ensure that LinearOperator's '@' gets called before NumPy's.
    __array_priority__ = 10.1

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

            first_notnone_index = 0
            while (
                first_notnone_index < len(row_block)
                and row_block[first_notnone_index] is None
            ):
                first_notnone_index += 1

            if first_notnone_index == len(row_block):
                raise ValueError("A matrix must not have row of None.")

            row_block_height = row_block[first_notnone_index].shape[0]
            self._nrows += row_block_height
            self._rowdims[i] = row_block_height

            for j, block in enumerate(row_block):
                if block is not None:
                    if block.shape[0] != row_block_height:
                        raise ValueError("Dimension mismatch in A.")
                    if self._coldims[j] is None:
                        self._coldims[j] = block.shape[1]
                        self._ncols += block.shape[1]
                    else:
                        if self._coldims[j] != block.shape[1]:
                            raise ValueError("Dimension mismatch in A.")

        if None in self._rowdims:
            raise ValueError("A matrix must not have row of None.")
        if None in self._coldims:
            raise ValueError("A matrix must not have column of None.")

        self._A = A
        self.shape = (self._nrows, self._ncols)
        self.T = _TransposedLinearOperator(self)

    def __matmul__(self, v):
        return self.matvec(v)

    def __rmatmul__(self, v):
        return self.rmatvec(v)

    def matvec(self, v):
        if v.shape[0] != self._ncols:
            raise ValueError("Dimension mismatch.")

        res = np.zeros(self._nrows)
        row_index = 0
        for i, row_block in enumerate(self._A):
            col_index = 0
            for j, block in enumerate(row_block):
                if block is not None:
                    if (
                        type(block) is sp.sparse.linalg._interface._CustomLinearOperator
                        or type(block) is LinearOperator
                    ):
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

        res = np.zeros(self._ncols)
        col_index = 0
        for i, row_block in enumerate(self._A):
            row_index = 0
            for j, block in enumerate(row_block):
                if block is not None:
                    if (
                        type(block) is sp.sparse.linalg._interface._CustomLinearOperator
                        or type(block) is LinearOperator
                    ):
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


def block_diag(linop_list):
    A = []
    num_blocks = len(linop_list)

    for i, block in enumerate(linop_list):
        inner_list = [None] * i
        inner_list.append(block)
        inner_list.extend([None] * (num_blocks - i - 1))
        A.append(inner_list)

    return LinearOperator(A)


def hstack(linop_list):
    A = [linop_list]
    return LinearOperator(A)


def vstack(linop_list):
    A = [[block] for block in linop_list]
    return LinearOperator(A)


class _TransposedLinearOperator(LinearOperator):
    __array_priority__ = 10.1

    def __init__(self, A):
        self.shape = (A.shape[1], A.shape[0])
        self.A = A
        return

    def __matmul__(self, v):
        return self.matvec(v)

    def __rmatmul__(self, v):
        return self.rmatvec(v)

    def matvec(self, v):
        return self.A.rmatvec(v)

    def rmatvec(self, v):
        return self.A.matvec(v)
