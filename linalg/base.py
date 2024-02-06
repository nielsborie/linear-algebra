class Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

class Vector:
    def __init__(self, data: list[float]) -> None:
        self.data = data
        self.shape = self._shape()

    def _assert_same_length(self, other: 'Vector') -> None:
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must have the same length")

    def __add__(self, other: 'Vector') -> 'Vector':
        self._assert_same_length(other)
        return Vector([x + y for x, y in zip(self.data, other.data)])

    def __sub__(self, other: 'Vector') -> 'Vector':
        self._assert_same_length(other)
        return Vector([x - y for x, y in zip(self.data, other.data)])

    def __mul__(self, other: 'Vector') -> 'Vector':
        self._assert_same_length(other)
        return Vector([x * y for x, y in zip(self.data, other.data)])

    def dot(self, other: 'Vector') -> float:
        self._assert_same_length(other)
        #return sum([self.data[i] * other.data[i] for i in range(len(self.data))])
        return Vector([x * y for x, y in zip(self.data, other.data)])

    def _shape(self) -> (int, int):
        return len(self.data), 1

    def scalar_product(self, scalar: Scalar) -> 'Vector':
        return Vector([scalar*self.data[i] for i in range(len(self.data))])

    def __repr__(self):
        return f"Vector({self.data})"

class Matrix:
    def __init__(self, data: list[list[float]]) -> None:
        self.num_rows: int = len(data[0])
        self.num_cols: int = len(data)
        self.data: list[list[float]] = data
        self._assert_shape()
        self.shape = self._shape()

    def _assert_same_num_rows(self, other: 'Matrix') -> None:
        if len(self.data) != len(other.data):
            raise ValueError("Matrices must have the same dimensions. Unexpected number of rows.")

    def _assert_same_num_cols(self, other: 'Matrix') -> None:
        if len(self.data[0]) != len(other.data[0]):
            raise ValueError("Matrices must have the same dimensions. Unexpected number of columns.")

    def _assert_shape(self) -> None:
        for i in range(self.num_rows):
            assert len(self.data[i]) == self.num_rows, "Inner row vectors must have the same number of elements"

    def _shape(self) -> (int, int):
        return self.num_rows, self.num_cols

    def __getitem__(self, key: tuple[int, int]) -> float:
        """
        Retrieve the value at the specified position in the matrix.

        __getitem__() is a magic method in Python, which when used in a class, allows its instances to use the [] (indexer) operators.
        Say x is an instance of this class, then x[i] is roughly equivalent to type(x).__getitem__(x, i).

        Parameters:
        - key (Tuple[int, int]): A tuple representing the row and column indices.

        Returns:
        - float: The value at the specified position in the matrix.
        """
        row, col = key
        if isinstance(row, int) and row >= self.num_rows:
            raise IndexError(f"Cannot retrieve the value of the line at index {row}, there are only {self.num_rows} rows")
        if isinstance(col, int) and col >= self.num_cols:
            raise IndexError(f"Cannot retrieve the value of the column at index {col}, there are only {self.num_cols} columns")
        return self.data[row][col]

    def __setitem__(self, key: tuple[int, int], value: float) -> None:
        """
        Set the value at the specified position in the matrix.

        Parameters:
        - key (Tuple[int, int]): A tuple representing the row and column indices.
        - value (float): The new value to set at the specified position.

        Returns:
        - None
        """
        row, col = key
        self.data[row][col] = value

    def __add__(self, other: 'Matrix') -> 'Matrix':
        self._assert_same_num_rows(other) or self._assert_same_num_cols(other)
        result = [[0 for _ in range(len(other.data[0]))] for _ in range(len(self.data))]
        for i in range(len(self.data)):
            for j in range(len(self.data[i])): #could be self.data[0] here
                result[i][j]: float = self.data[i][j] + other.data[i][j]
        return Matrix(result)
        #return Matrix([[self.data[i][j] + other.data[i][j] for j in range(len(self.data[0]))] for i in range(len(self.data))])

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        self._assert_same_num_rows(other) or self._assert_same_num_cols(other)
        result = [[0 for _ in range(len(other.data[0]))] for _ in range(len(self.data))]
        for i in range(len(self.data)):
            for j in range(len(self.data[i])): #could be self.data[0] here
                result[i][j]: float = self.data[i][j] - other.data[i][j]
        return Matrix(result)
        #return Matrix([[self.data[i][j] - other.data[i][j] for j in range(len(self.data[0]))] for i in range(len(self.data))])

    def __mul__(self, other: 'Matrix') -> 'Matrix':
        if len(self.data[0]) != len(other.data):
            raise ValueError("The number of columns in the matrix must be equal to the number of rows in the other matrix.")
        result = [[0 for _ in range(len(other.data[0]))] for _ in range(len(self.data))]
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                for k in range(len(other.data)):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        return Matrix(result)

    def __repr__(self):
        return "\n".join([" ".join(map(str, row)) for row in self.data])

    def dot(self, matrix: 'Matrix') -> 'Matrix':
        result: list[list[float]] = [[0] * matrix.num_cols for _ in range(self.num_rows)]
        for i in range(self.num_rows):
            vector1_row = Vector(self.data[i])
            for j in range(matrix.num_cols):
                vector2_col = Vector([matrix.data[k][j] for k in range(matrix.num_rows)])
                result[i][j] = vector1_row.dot(vector2_col)
        return Matrix(result)

    def transpose(self) -> 'Matrix':
        result: list[list[float]] = [[0] * self.num_cols for _ in range(self.num_rows)]
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                result[i][j] = self.data[j][i]
        return Matrix(result)

    def transpose2(self) -> 'Matrix':
        transposed_data = [[row[i] for row in self.data] for i in range(self.num_cols)]
        return Matrix(transposed_data)

if __name__ == '__main__':
    matrix1 = Matrix([[1, 2],
                      [3, 4]])
    matrix2 = Matrix([[1, 2],
                      [3, 4]])
    matrix3 = matrix1 * matrix2
    print(f"A :")
    print(matrix1)

    print(f"B :")
    print(matrix2)

    print(f"C = A*B :")
    print(matrix3)

    print(f"C = A.dot(B) :")
    print(matrix1.dot(matrix2))
    #print(f"{matrix2}")
    #print(f"{matrix3}")

    A = Matrix([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]])
    print('Transpose : ')
    print('A : ')
    print(A)
    print('A.T : ')
    print(A.transpose())
    print('A.T (bis): ')
    print(A.transpose2())
    print(f"{A[1, 3]=}")
    print(f"{A[:, 3]=}")
    print(f"{A[1, 3]=}")
    #print(f"{A[:, 4]=}")