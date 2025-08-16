import numpy as np
from scipy.integrate import solve_ivp
import Models
import torch

def equation_of_motion(time, state, gamma, length, radius):
    phi, phi_dot = state

    phi_ddot = ((radius * (gamma ** 2)) / length) * np.cos(phi - gamma * time) - (9.81) * np.sin(phi) / length

    return [phi_dot, phi_ddot]

def Lagrangian_Solver(phi_intial, phi_dot_initial, gamma, length, radius):
    state = [phi_intial, phi_dot_initial]

    t_span = (0, 50)
    t_eval = np.linspace(t_span[0], t_span[1], 5000)

    sol = solve_ivp(equation_of_motion, t_span, state, args=( gamma, length, radius),
                    t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-10)

    return (sol.t, sol.y)


def rk4(func, state, time, dt, lnn):
    k1 = func(state, time, lnn)
    k2 = func(state + 0.5 * dt * k1, state + 0.5 * dt, lnn)
    k3 = func(state + 0.5 * dt * k2, state + 0.5 * dt, lnn)
    k4 = func(state + dt * k3, state + dt, lnn)
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


