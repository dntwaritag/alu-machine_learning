#!/usr/bin/env python3
"""
Module for calculating the inverse of a given matrix.
"""


def determinant(matrix):
    """
    Calculate the determinant of a matrix.

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
    Calculate the cofactor matrix of a matrix.

    Args:
        matrix (list of lists): The matrix whose cofactor matrix
        should be calculated.

    Returns:
        list of lists: The cofactor matrix of the given matrix.
    """
    if len(matrix) == 1 and len(matrix[0]) == 1:  # Single element
        return [[1]]

    cofactor_matrix = []
    for i in range(len(matrix)):
        cofactor_row = []
        for j in range(len(matrix)):
            minor_matrix = [row[:j] + row[j + 1:] for k, row in enumerate(matrix)
                            if k != i]
            minor_det = determinant(minor_matrix)
            cofactor_value = ((-1) ** (i + j)) * minor_det
            cofactor_row.append(cofactor_value)
        cofactor_matrix.append(cofactor_row)
    return cofactor_matrix


def adjugate(matrix):
    """
    Calculate the adjugate matrix of a matrix.

    Args:
        matrix (list of lists): The matrix whose adjugate matrix
        should be calculated.

    Returns:
        list of lists: The adjugate matrix of the given matrix.
    """
    cofactor_matrix = cofactor(matrix)
    # Transpose the cofactor matrix to get the adjugate
    adjugate_matrix = list(map(list, zip(*cofactor_matrix)))
    return adjugate_matrix


def inverse(matrix):
    """
    Calculate the inverse of a matrix.

    Args:
        matrix (list of lists): The matrix whose inverse should be
        calculated.

    Returns:
        list of lists: The inverse of the matrix, or None if the
        matrix is singular.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None  # Matrix is singular, no inverse exists

    adjugate_matrix = adjugate(matrix)
    inverse_matrix = [[elem / det for elem in row] for row in adjugate_matrix]
    return inverse_matrix
