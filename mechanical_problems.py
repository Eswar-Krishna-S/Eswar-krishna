"""
NumPy Universal Functions - Mechanical Engineering Problems
Complete all functions using NumPy ufuncs (no loops!)

Student Name: Eswar Krishna
Student ID: 715523114014
"""

import numpy as np


def von_mises_stress(sigma1, sigma2):
    """
    Calculate von Mises stress from principal stresses.

    Formula: sigma_vm = sqrt(sigma1^2 - sigma1*sigma2 + sigma2^2)
    """
    sigma1 = np.asarray(sigma1)
    sigma2 = np.asarray(sigma2)
    return np.sqrt(sigma1**2 - sigma1 * sigma2 + sigma2**2)


def projectile_trajectory(v0, angles, t):
    """
    Calculate projectile x, y coordinates for multiple angles.

    x = v0 * cos(theta) * t
    y = v0 * sin(theta) * t - 0.5 * g * t^2
    """
    g = 9.81  # m/s^2
    angles = np.asarray(angles)
    t = np.asarray(t)
    theta = np.deg2rad(angles)

    x = v0 * np.cos(theta)[:, None] * t[None, :]
    y = v0 * np.sin(theta)[:, None] * t[None, :] - 0.5 * g * (t[None, :] ** 2)
    return x, y


def force_resultant(fx, fy):
    """
    Calculate magnitude and direction of resultant forces.
    """
    fx = np.asarray(fx)
    fy = np.asarray(fy)
    magnitude = np.hypot(fx, fy)
    angle_degrees = np.degrees(np.arctan2(fy, fx))
    return magnitude, angle_degrees


def thermal_expansion(L0, alpha, delta_T):
    """
    Calculate length change due to thermal expansion.

    delta_L = alpha * L0 * delta_T
    """
    L0 = np.asarray(L0)
    alpha = np.asarray(alpha)
    delta_T = np.asarray(delta_T)
    delta_L = alpha * L0 * delta_T
    L_final = L0 + delta_L
    return delta_L, L_final


def angular_velocity_conversion(rpm):
    """
    Convert RPM to rad/s and calculate angular displacement after 5 seconds.

    omega (rad/s) = RPM * (2*pi / 60)
    theta = omega * t
    """
    t = 5  # seconds
    rpm = np.asarray(rpm)
    omega = rpm * (2 * np.pi / 60.0)
    theta = omega * t
    return omega, theta


def beam_deflection(x, L, w, E, I):
    """
    Calculate deflection of simply supported beam with uniform load.

    y = (w/(24*E*I)) * x * (L^3 - 2*L*x^2 + x^3)
    """
    x = np.asarray(x)
    coef = w / (24.0 * E * I)
    return coef * x * (L**3 - 2.0 * L * x**2 + x**3)


def velocity_components(velocities, angles):
    """
    Resolve velocities into x and y components.
    """
    velocities = np.asarray(velocities)
    angles = np.deg2rad(np.asarray(angles))
    vx = velocities * np.cos(angles)
    vy = velocities * np.sin(angles)
    return vx, vy


def power_to_torque(power, omega):
    """
    Calculate torque from power and angular velocity.

    tau = P / omega
    """
    power = np.asarray(power)
    omega = np.asarray(omega)
    return power / omega


def spring_system(k, x):
    """
    Calculate spring force and potential energy.

    F = -k * x
    U = 0.5 * k * x^2
    """
    k = np.asarray(k)
    x = np.asarray(x)
    force = -k * x
    potential = 0.5 * k * x**2
    return force, potential


def damped_oscillation(A, b, omega, t):
    """
    Calculate displacement for damped harmonic motion.

    x(t) = A * exp(-b*t) * cos(omega*t)
    """
    t = np.asarray(t)
    return A * np.exp(-b * t) * np.cos(omega * t)