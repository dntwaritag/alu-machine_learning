#!/usr/bin/env python3
"""
Module for calculating the cofactor matrix of a given matrix.
"""


def determinant(matrix):
    """
    Helper function to calculate the determinant of a matrix.

    Args:
        matrix (list of lists): The matrix to calculate the determinant for.

    Returns:
        int or float: The determinant of the matrix.
    """
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(len(matrix)):
        submatrix = [row[:col] + row[col + 1:] for row in matrix[1:]]
        det += ((-1) ** col) * matrix[0][col] * determinant(submatrix)
    return det


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix.

    Args:
        matrix (list of lists): The matrix whose cofactor matrix
        should be calculated.

    Returns:
        list of lists: The cofactor matrix of the given matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1 and len(matrix[0]) == 0:  # Handling empty matrix
        raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix)):
            submatrix = [row[:j] + row[j + 1:] for k, row in enumerate(matrix)
                         if k != i]
            # Apply (-1)^(i+j) to adjust the sign
            cofactor_value = ((-1) ** (i + j)) * determinant(submatrix)
            minor_row.append(cofactor_value)
        minor_matrix.append(minor_row)

    # Correcting for a single-element matrix to return [[1]] instead of [[0]]
    if len(matrix) == 1:
        return [[1]]

    return minor_matrix
