import numpy as np
import time
import matplotlib.pyplot as plt

# =========================================================
# Functions and Gradients
# =========================================================

def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    dy = 200*(x[1] - x[0]**2)
    return np.array([dx, dy])


def sin_inv(x):
    if abs(x) < 1e-6:
        return 0.0
    return np.sin(1/x)

def grad_sin_inv(x):
    if abs(x) < 1e-6:
        return 0.0
    return -np.cos(1/x) / (x**2)

# =========================================================
# Optimizers
# =========================================================

def gradient_descent(grad, x0, lr, tol=1e-6, max_iter=5000):
    x = x0.copy()
    history = []
    start = time.time()

    for _ in range(max_iter):
        g = grad(x)
        x_new = x - lr * g
        history.append(np.linalg.norm(g))
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    return x, history, time.time() - start


def sgd_momentum(grad, x0, lr, beta=0.9, tol=1e-6, max_iter=5000):
    x = x0.copy()
    v = np.zeros_like(x)
    history = []
    start = time.time()

    for _ in range(max_iter):
        g = grad(x)
        v = beta * v + lr * g
        x_new = x - v
        history.append(np.linalg.norm(g))
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    return x, history, time.time() - start


def adagrad(grad, x0, lr, eps=1e-8, tol=1e-6, max_iter=5000):
    x = x0.copy()
    G = np.zeros_like(x)
    history = []
    start = time.time()

    for _ in range(max_iter):
        g = grad(x)
        G += g**2
        x_new = x - lr * g / (np.sqrt(G) + eps)
        history.append(np.linalg.norm(g))
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    return x, history, time.time() - start


def rmsprop(grad, x0, lr, beta=0.9, eps=1e-8, tol=1e-6, max_iter=5000):
    x = x0.copy()
    E = np.zeros_like(x)
    history = []
    start = time.time()

    for _ in range(max_iter):
        g = grad(x)
        E = beta * E + (1 - beta) * g**2
        x_new = x - lr * g / (np.sqrt(E) + eps)
        history.append(np.linalg.norm(g))
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    return x, history, time.time() - start


def adam(grad, x0, lr, b1=0.9, b2=0.999, eps=1e-8, tol=1e-6, max_iter=5000):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = []
    start = time.time()

    for t in range(1, max_iter + 1):
        g = grad(x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g**2)
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        x_new = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(np.linalg.norm(g))
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    return x, history, time.time() - start

# =========================================================
# Experiments
# =========================================================

optimizers = {
    "GD": gradient_descent,
    "SGD+Momentum": sgd_momentum,
    "Adagrad": adagrad,
    "RMSprop": rmsprop,
    "Adam": adam
}

learning_rates = [0.01, 0.05, 0.1]

# ------------------------------
# Rosenbrock Function
# ------------------------------

x0_rosen = np.array([-1.5, 1.5])

for lr in learning_rates:
    plt.figure(figsize=(10,6))
    print(f"\nRosenbrock | Learning Rate = {lr}")
    for name, opt in optimizers.items():
        x_opt, history, t = opt(grad_rosenbrock, x0_rosen, lr)
        print(f"{name}: x = {x_opt}, f(x) = {rosenbrock(x_opt):.6f}, time = {t:.4f}s")
        plt.plot(history, label=name)

    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Gradient Norm")
    plt.title(f"Rosenbrock Convergence (α={lr})")
    plt.legend()
    plt.show()

# ------------------------------
# sin(1/x) Function
# ------------------------------

x0_sin = np.array(0.3)

def grad_wrapper(x):
    return grad_sin_inv(x)

for lr in learning_rates:
    plt.figure(figsize=(10,6))
    print(f"\nsin(1/x) | Learning Rate = {lr}")
    for name, opt in optimizers.items():
        x_opt, history, t = opt(grad_wrapper, x0_sin, lr)
        print(f"{name}: x = {x_opt}, f(x) = {sin_inv(x_opt):.6f}, time = {t:.4f}s")
        plt.plot(history, label=name)

    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Gradient Norm")
    plt.title(f"sin(1/x) Convergence (α={lr})")
    plt.legend()
    plt.show()
