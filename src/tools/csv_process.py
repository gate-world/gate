import numpy as np
import csv


def save_matrix_to_csv(matrix, filename):
    """
    Save a NumPy matrix to a CSV file.

    Parameters:
    matrix (np.ndarray): The NumPy array to save.
    filename (str): The path to the CSV file.
    """
    try:
        np.savetxt(filename, matrix, delimiter=',', fmt='%g')
        # print(f"Matrix saved to {filename} successfully.")
    except Exception as e:
        print(f"An error occurred while saving the matrix: {e}")


def read_csv_to_matrix(filename):
    """
    Read a CSV file into a NumPy matrix.

    Parameters:
    filename (str): The path to the CSV file.

    Returns:
    np.ndarray: The NumPy array containing the data from the CSV file.
    """
    try:
        matrix = np.loadtxt(filename, delimiter=',')
        # print(f"Matrix loaded from {filename} successfully.")
        return matrix
    except Exception as e:
        print(f"An error occurred while loading the matrix: {e}")
        return None


# Example usage:
if __name__ == "__main__":
    # Create a sample matrix
    sample_matrix = np.array([[1.5, 2.3, 3.1], [4.1, 5.2, 6.3], [7.4, 8.5, 9.6]])

    # Save the matrix to a CSV file
    save_matrix_to_csv(sample_matrix, 'matrix.csv')

    # Read the matrix back from the CSV file
    loaded_matrix = read_csv_to_matrix('matrix.csv')
    print("Loaded Matrix:")
    print(loaded_matrix)
