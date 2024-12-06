import numpy as np
import matplotlib.pyplot as plt

def compute_capacity(channel_matrix, epsilon=1e-6, max_iterations=1000):
    num_inputs, num_outputs = channel_matrix.shape

    # Check if channel matrix rows sum to 1
    if not np.allclose(channel_matrix.sum(axis=1), 1):
        raise ValueError("Each row of the channel matrix must sum to 1.")
    
    # Initialize uniform distribution for input symbols
    P_X = np.full(num_inputs, 1 / num_inputs)
    log_channel_matrix = np.log(np.maximum(channel_matrix, 1e-12))  # Small value to avoid log(0)

    for i in range(max_iterations):
        # Calculate output distribution Q based on P_X
        Q = P_X @ channel_matrix
        # Calculate divergence for each input symbol
        divergence = np.zeros(num_inputs)
        for x in range(num_inputs):
            for y in range(num_outputs):
                if channel_matrix[x, y] > 0 and Q[y] > 0:
                    divergence[x] += channel_matrix[x, y] * (log_channel_matrix[x, y] - np.log(Q[y]))
        
        # Update P_X using the computed divergences
        P_X_new = P_X * np.exp(divergence)
        P_X_new /= np.sum(P_X_new)  # Normalize so that sum of P_X_new equals 1
        
        # Check for convergence
        if np.linalg.norm(P_X_new - P_X, ord=1) < epsilon:
            P_X = P_X_new
            break
        
        P_X = P_X_new  # Update P_X for the next iteration

    # Final output distribution based on optimal P_X
    Q = P_X @ channel_matrix
    
    # Calculate the channel capacity C = I(X; Y)
    capacity = 0.0
    for x in range(num_inputs):
        for y in range(num_outputs):
            if channel_matrix[x, y] > 0 and Q[y] > 0:
                capacity += P_X[x] * channel_matrix[x, y] * (np.log(channel_matrix[x, y]) - np.log(Q[y]))
    
    # Convert capacity from natural units to bits
    capacity /= np.log(2)
    
    return capacity, P_X

def load_channel_matrix(filename):
    """Load the channel matrix from a file."""
    return np.loadtxt(filename)

def main():
    # Define channel files
    channel_files = {
        "A": "channel_A.txt",
        "B": "channel_B.txt",
        "C": "channel_C.txt",
        "D": "channel_D.txt",
        "E": "channel_E.txt"
    }

    # Compute and print capacities for each channel
    for name, file in channel_files.items():
        matrix = load_channel_matrix(file)
        capacity, optimal_P_X = compute_capacity(matrix)
        print(f"Capacity of Channel {name}: {capacity:.4f} bits")
        # print(f"Optimal Input Distribution for Channel {name}: {optimal_P_X}\n")

if __name__ == "__main__":
    main()
