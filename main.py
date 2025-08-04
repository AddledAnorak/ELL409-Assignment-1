import numpy as np
from scipy.optimize import linprog
from itertools import product

def is_linearly_separable(X, y):
    num_samples, num_features = X.shape

    num_variables = num_features + 1

    c = np.zeros(num_variables)

    A_ub = np.zeros((num_samples, num_variables))
    b_ub = -np.ones(num_samples)

    for i in range(num_samples):
        current_X = X[i, :]
        current_y = y[i]

        yi_scaled = 1 if current_y == 1 else -1

        A_ub_row = np.zeros(num_variables)
        A_ub_row[:num_features] = current_X * (-yi_scaled)
        A_ub_row[num_features] = -yi_scaled

        A_ub[i, :] = A_ub_row

    bounds = [(None, None)] * num_variables

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    return res.success


def count_linearly_separable_boolean_functions(N):
    input_vectors = np.array(list(product([0, 1], repeat=N)))
    num_input_vectors = len(input_vectors)

    total_boolean_functions = 2**num_input_vectors

    separable_count = 0
    for i in range(total_boolean_functions):
        binary_outputs = bin(i)[2:].zfill(num_input_vectors)
        current_y = np.array([int(bit) for bit in binary_outputs])

        if is_linearly_separable(input_vectors, current_y):
            separable_count += 1

    return separable_count



print(f"For N=2: {count_linearly_separable_boolean_functions(2)}")
print(f"For N=3: {count_linearly_separable_boolean_functions(3)}")
print(f"For N=4: {count_linearly_separable_boolean_functions(4)}")
