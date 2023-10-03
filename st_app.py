import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def f(r, u, theta, n):
    return -theta**n - 2*u/r


def analytical(t, n):
    if n == 0:
        return 1 - t**2/6
    elif n == 1:
        return np.sin(t)/t
    elif n == 5:
        return (1 + t**2/3)**(-0.5)



def euler_method(f, a, b, N, IV, V, polytropic_index):
    # Determine step size
    h = (b - a) / float(N)
    t = np.arange(a, b + h, h)

    theta = np.zeros(N+1)
    dtheta = np.zeros(N+1)

    theta[0], dtheta[0] = IV

    # Apply Euler's method
    for i in range(1, N+1):
        theta[i] = theta[i-1] + h * dtheta[i-1]
        dtheta[i] = dtheta[i-1] + h * \
            f(a + i*h, dtheta[i-1], theta[i-1], polytropic_index)

    return t, theta, h


def heuns_method(f, a, b, N, IV, V, polytropic_index):

    h = (b - a) / float(N)

    t = np.arange(a, b + h, h)

    theta = np.zeros(N+1)
    dtheta = np.zeros(N+1)

    theta[0], dtheta[0] = IV

    for i in range(1, N+1):
        theta_half = theta[i-1] + h * dtheta[i-1]
        dtheta_half = dtheta[i-1] + h * \
            f(a + i*h, dtheta[i-1], theta[i-1], polytropic_index)

        theta[i] = theta[i-1] + 0.5 * h * (dtheta[i-1] + dtheta_half)
        dtheta[i] = dtheta[i-1] + 0.5 * h * (f(a + i*h, dtheta[i-1], theta[i-1], polytropic_index) +
                                             f(a + (i+1)*h, dtheta_half, theta[i], polytropic_index))

    return t, theta, h


def runge_kutta_method(f, a, b, N, IV, V, polytropic_index):
    # Determine step size
    h = (b - a) / float(N)

    t = np.arange(a, b + h, h)

    theta = np.zeros(N+1)
    dtheta = np.zeros(N+1)

    # Set initial values
    theta[0], dtheta[0] = IV

    for i in range(1, N+1):
        k1 = h * dtheta[i-1]
        l1 = h * f(a + i*h, dtheta[i-1], theta[i-1], polytropic_index)

        k2 = h * (dtheta[i-1] + l1/2)
        l2 = h * f(a + i*h + h/2, dtheta[i-1] +
                   k1/2, theta[i-1] + l1/2, polytropic_index)

        k3 = h * (dtheta[i-1] + l2/2)
        l3 = h * f(a + i*h + h/2, dtheta[i-1] +
                   k2/2, theta[i-1] + l2/2, polytropic_index)

        k4 = h * (dtheta[i-1] + l3)
        l4 = h * f(a + i*h + h, dtheta[i-1] + k3,
                   theta[i-1] + l3, polytropic_index)

        theta[i] = theta[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
        dtheta[i] = dtheta[i-1] + (l1 + 2*l2 + 2*l3 + l4) / 6

    return t, theta, h,dtheta


st.title("Lane-Emden Equation Solver")

polytropic_index = st.sidebar.slider(
    "Polytropic Index (n)", min_value=0.0, max_value=5.0, step=0.5, value=1.0)
N = st.sidebar.slider("Number of Steps (N)", min_value=10,
                      max_value=5000, step=10, value=1000)

a = 0.0
b = 40.0

t, theta_euler, h_euler = euler_method(
    f, a, b, N, (1.0, 0.0), (0.0, 0.0), polytropic_index)
t, theta_heun, h_heun = heuns_method(
    f, a, b, N, (1.0, 0.0), (0.0, 0.0), polytropic_index)
t, theta_rk4, h_rk4,dtheta_rk4 = runge_kutta_method(
    f, a, b, N, (1.0, 0.0), (0.0, 0.0), polytropic_index)

root_index = np.argmax(theta_rk4 < 0)  # Assuming the root is where theta_rk4 crosses zero
root_x = t[root_index]
# st.write(f"θ Root Value: {root_x:.2f}")
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, theta_euler, label="Euler's Method")
plt.plot(t, theta_heun, label="Heun's Method")
plt.plot(t, theta_rk4, label="Runge-Kutta Method")


if polytropic_index in [0, 1, 5]:
    t_analytical = np.linspace(a, b, 1000)
    theta_analytical = analytical(t_analytical, polytropic_index)
    plt.plot(t_analytical, theta_analytical,
             label=f"Analytical (N={polytropic_index})", linestyle="--")
    
if np.any(theta_rk4 < 0):
    root_x = t[root_index]
    plt.plot(root_x, 0, 'rx', markersize=10, label='Root')

    st.write(f"θ Root Value: {root_x:.2f}")
else:
    st.write("No root found")




plt.title("Lane-Emden Equation Solutions")
plt.xlabel("ξ")
plt.ylabel("θ")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, dtheta_rk4, label="Runge-Kutta Method")
plt.xlabel("ξ")
plt.ylabel("dθ/dξ")
plt.grid(True)
root_index_1 = np.argmax(theta_rk4 < 0)  
# root_x = t[root_index_1]

if np.any(theta_rk4 < 0):
    root_x = t[root_index]
    plt.plot(root_x, dtheta_rk4[root_index], 'rx', markersize=10, label='Root')
    st.write(f"dθ/dξ Root Value: {dtheta_rk4[root_index]:.2f}")
    st.write(f"-ξ^2 * dθ/dξ Root Value: {-root_x**2 * dtheta_rk4[root_index]:.2f}")
else:
    st.write("No root found")

# st.write(f"dθ Root Value: {root_x:.2f}")
# plt.plot(root_x, 0, 'rx', markersize=10, label='Root')

st.pyplot(plt)
