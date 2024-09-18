#!/usr/bin/env python3
"""
Module to calculate the determinant of a matrix.
"""


def determinant(matrix):
    """
    Calculate the determinant of a matrix. Validates that the input is a
    list of lists and checks for squareness. Handles 0x0 matrices by
    returning 1, in accordance with mathematical conventions.

    Args:
        matrix (list of lists): The matrix to calculate the determinant for.

    Returns:
        int or float: The determinant of the matrix, with special handling for
        0x0 matrices.

    Raises:
        TypeError: If the matrix is not a list of lists.
        ValueError: If the matrix is not square.
    """
    # Validate matrix format
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    num_rows = len(matrix)

    # Early return for 0x0 matrix
    if num_rows == 0:
        return 1

    # Validate square shape
    if any(len(row) != num_rows for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case for 1x1 matrix
    if num_rows == 1:
        return matrix[0][0]

    # Recursive case for matrices larger than 1x1
    det = 0
    for col in range(num_rows):
        minor = [row[:col] + row[col + 1:] for row in matrix[1:]]
        cofactor = (-1) ** col * matrix[0][col] * determinant(minor)
        det += cofactor

    return det
