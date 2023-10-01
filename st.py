import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define the Lane-Emden equation
def f(r, u, theta, n):
    return -theta**n - 2*u/r

# Analytical solutions
def analytical(t, n):
    if n == 0:
        return 1 - t**2/6
    elif n == 1:
        return np.sin(t)/t
    elif n == 5:
        return (1 + t**2/3)**(-0.5)

# Euler's Method
def euler_method(f, a, b, N, IV, V, polytropic_index):
    # Determine step size
    h = (b - a) / float(N)

    # Create mesh
    t = np.arange(a, b + h, h)

    # Initialize arrays for theta and dtheta
    theta = np.zeros(N+1)
    dtheta = np.zeros(N+1)

    # Set initial values
    theta[0], dtheta[0] = IV

    # Apply Euler's method
    for i in range(1, N+1):
        theta[i] = theta[i-1] + h * dtheta[i-1]
        dtheta[i] = dtheta[i-1] + h * f(a + i*h, dtheta[i-1], theta[i-1], polytropic_index)

    return t, theta

# Multistep Method
def multistep_method(f, a, b, N, IV, V, polytropic_index):
    # Determine step size
    h = (b - a) / float(N)

    # Create mesh
    t = np.arange(a, b + h, h)

    # Initialize arrays for theta and dtheta
    theta = np.zeros(N+1)
    dtheta = np.zeros(N+1)

    # Set initial values
    theta[0], dtheta[0] = IV

    # Apply Multistep method
    for i in range(1, N+1):
        theta_half = theta[i-1] + (h/2) * dtheta[i-1]
        dtheta_half = dtheta[i-1] + (h/2) * f(a + i*h, dtheta[i-1], theta[i-1], polytropic_index)

        theta[i] = theta[i-1] + (h/2) * (dtheta[i-1] + dtheta_half)
        dtheta[i] = dtheta[i-1] + (h/2) * (f(a + i*h, dtheta[i-1], theta[i-1], polytropic_index) +
                                           f(a + (i+1)*h, dtheta_half, theta[i], polytropic_index))

    return t, theta

# Runge-Kutta Method
def runge_kutta_method(f, a, b, N, IV, V, polytropic_index):
    # Determine step size
    h = (b - a) / float(N)

    # Create mesh
    t = np.arange(a, b + h, h)

    # Initialize arrays for theta and dtheta
    theta = np.zeros(N+1)
    dtheta = np.zeros(N+1)

    # Set initial values
    theta[0], dtheta[0] = IV

    # Apply Runge-Kutta method
    for i in range(1, N+1):
        k1 = h * dtheta[i-1]
        l1 = h * f(a + i*h, dtheta[i-1], theta[i-1], polytropic_index)

        k2 = h * (dtheta[i-1] + l1/2)
        l2 = h * f(a + i*h + h/2, dtheta[i-1] + k1/2, theta[i-1] + l1/2, polytropic_index)

        k3 = h * (dtheta[i-1] + l2/2)
        l3 = h * f(a + i*h + h/2, dtheta[i-1] + k2/2, theta[i-1] + l2/2, polytropic_index)

        k4 = h * (dtheta[i-1] + l3)
        l4 = h * f(a + i*h + h, dtheta[i-1] + k3, theta[i-1] + l3, polytropic_index)

        theta[i] = theta[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
        dtheta[i] = dtheta[i-1] + (l1 + 2*l2 + 2*l3 + l4) / 6

    return t, theta

# Streamlit UI
st.title("Lane-Emden Equation Solver")

# Sidebar for polytropic index
polytropic_index = st.sidebar.slider("Polytropic Index (n)", min_value=0.0, max_value=5.0, step=0.5, value=1.0)

# Calculate solutions using Euler's method
a = 0
b = 10
N = 1000

t, theta_euler = euler_method(f, a, b, N, (1.0, 0.0), (0.0, 0.0), polytropic_index)
t, theta_multistep = multistep_method(f, a, b, N, (1.0, 0.0), (0.0, 0.0), polytropic_index)
t, theta_runge_kutta = runge_kutta_method(f, a, b, N, (1.0, 0.0), (0.0, 0.0), polytropic_index)

# Plot the solutions
plt.figure(figsize=(10, 6))
plt.plot(t, theta_euler, label="Euler's Method")
plt.plot(t, theta_multistep, label="Multistep Method")
plt.plot(t, theta_runge_kutta, label="Runge-Kutta Method")

# Check if the selected index matches one of the analytical solutions and plot it
if polytropic_index in [0, 1, 5]:
    t_analytical = np.linspace(0, 10, 1000)
    theta_analytical = analytical(t_analytical, polytropic_index)
    plt.plot(t_analytical, theta_analytical, label=f"Analytical (N={polytropic_index})", linestyle="--")

plt.title("Lane-Emden Equation Solutions")
plt.xlabel("ξ")
plt.ylabel("θ")
plt.legend()
st.pyplot(plt)
