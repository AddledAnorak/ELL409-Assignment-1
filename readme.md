# Linearly Separable Boolean Functions Counter

This repository contains Python code to determine the number of **linearly separable boolean functions** for a given N-dimensional input space. Specifically, it's configured to calculate this for a 2D, 3D and 4D input space.

---

## What is a Linearly Separable Boolean Function?

A **boolean function** takes binary inputs (0s and 1s) and produces a binary output (0 or 1). For an N-dimensional input space, there are $2^N$ possible input vectors. For example, in a 2D space, the inputs are (0,0), (0,1), (1,0), (1,1).

A boolean function is considered **linearly separable** if its output can be perfectly predicted by a simple linear model. This means there exists a **hyperplane** (a line in 2D, a plane in 3D, or a higher-dimensional equivalent) that can separate the input points where the function outputs '1' from the input points where it outputs '0'.

Mathematically, for an input vector $x = (x_1, x_2, \dots, x_N)$, a function $f(x)$ is linearly separable if we can find weights $w = (w_1, w_2, \dots, w_N)$ and a bias $b$ such such that:

* If $f(x) = 1$, then $w_1 x_1 + w_2 x_2 + \dots + w_N x_N + b \ge \epsilon$

* If $f(x) = 0$, then $w_1 x_1 + w_2 x_2 + \dots + w_N x_N + b \le -\epsilon$

Here, $\epsilon$ is a small positive margin (we can use $\epsilon=1$ for simplicity in feasibility checking).

---

## Formulating Separability as a Linear Programming Problem

The problem of finding $w$ and $b$ that satisfy the above conditions can be formulated as a **Linear Programming (LP) feasibility problem**. An LP feasibility problem seeks to find *any* solution that satisfies a set of linear inequalities, rather than optimizing an objective function.

Let's transform the conditions:

1. For outputs $y_i = 1$: $w \cdot x_i + b \ge 1$

2. For outputs $y_i = 0$: $w \cdot x_i + b \le -1$

To fit the standard `linprog` format ($A_{ub}x \le b_{ub}$), we can multiply the first inequality by -1:

1. For outputs $y_i = 1$: $- (w \cdot x_i + b) \le -1$

2. For outputs $y_i = 0$: $w \cdot x_i + b \le -1$

A more unified approach, often seen in perceptron learning, is to scale the output $y_i$ to be either $1$ (for original $y_i=1$) or $-1$ (for original $y_i=0$). Let's call this scaled output $y'_i$. Then, the condition for all points becomes:

$y'_i \cdot (w \cdot x_i + b) \ge 1$

Again, to convert to $A_{ub}x \le b_{ub}$ form, we multiply by -1:

$-y'_i \cdot (w \cdot x_i + b) \le -1$

Here, our variables are the weights $w_1, \dots, w_N$ and the bias $b$. If a solution exists for these inequalities, the function is linearly separable. If no solution exists, it is not.

---

## How `scipy.optimize.linprog` is Used

The `scipy.optimize.linprog` function is a powerful tool for solving linear programming problems. While it's typically used for optimization (minimizing or maximizing an objective function subject to constraints), it can also be used to check for feasibility.

In this code:

* **Objective Function (`c`):** Since we only care about feasibility, the objective function is set to minimize 0 (i.e., `c = np.zeros(num_variables)`). This means we're not trying to find the "best" separating hyperplane, just *any* separating hyperplane.

* **Inequality Constraints (`A_ub`, `b_ub`):**

  * `A_ub` is a matrix where each row corresponds to one of the $-y'_i \cdot (w \cdot x_i + b)$ expressions. The columns correspond to the coefficients of $w_1, \dots, w_N, b$.

  * `b_ub` is a vector where each element is the right-hand side of the inequality, which is `-1` for all constraints.

* **Bounds (`bounds`):** The weights and bias can be any real number, so the bounds are set to `(None, None)` for all variables, indicating no lower or upper limit.

* **Method (`method='highs'`):** The 'highs' solver is a robust and efficient method for linear programming.

The `linprog` function returns a `scipy.optimize.OptimizeResult` object. If `res.success` is `True`, it indicates that a feasible solution was found, meaning the boolean function is linearly separable.

---

## Code Structure and Functions

The code is organized into two main functions:

### `is_linearly_separable(X, y)`

This function takes a set of input vectors `X` and their corresponding boolean outputs `y` and determines if the function they represent is linearly separable.

* **Inputs:**

  * `X` (NumPy array): A 2D array where each row is an input vector (e.g., `[0, 1, 0, 1]`).

  * `y` (NumPy array): A 1D array of corresponding binary outputs (0 or 1).

* **Process:**

  1. It constructs the `A_ub` matrix and `b_ub` vector based on the linear programming formulation described above.

  2. It calls `scipy.optimize.linprog` with these constraints.

  3. It returns `True` if `linprog` finds a feasible solution (`res.success` is `True`), indicating separability, and `False` otherwise.

### `count_linearly_separable_boolean_functions(N)`

This is the main function that orchestrates the counting process.

* **Input:**

  * `N` (int): The dimensionality of the input space

* **Process:**

  1. **Generates all input vectors:** For a given `N`, it creates all $2^N$ possible binary input vectors (e.g., for $N=2$, it generates `(0,0), (0,1), (1,0), (1,1)`).

  2. **Iterates through all boolean functions:** There are $2^{(2^N)}$ possible boolean functions for an N-dimensional input space. The code iterates through each of these functions. Each function is represented by a unique binary string of length $2^N$, where each bit corresponds to the output (0 or 1) for one of the $2^N$ input vectors.

  3. **Checks separability:** For each boolean function, it calls `is_linearly_separable()` to determine if it can be separated.

  4. **Counts and reports:** It keeps a running count of separable functions and prints progress updates.

* **Output:** The total count of linearly separable boolean functions for the specified `N`.

---

## Running the Code

To run the code, simply execute the Python script. It will automatically start the computation for `N=2, 3, 4`.