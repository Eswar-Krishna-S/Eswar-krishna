"""
NumPy Universal Functions - Mechanical Engineering Problems
Complete all functions using NumPy ufuncs (no loops!)

Student Name: ___________________
Student ID: ___________________
"""

import numpy as np


def von_mises_stress(sigma1, sigma2):
    """
    Calculate von Mises stress from principal stresses.
    
    Formula: sigma_vm = sqrt(sigma1^2 - sigma1*sigma2 + sigma2^2)
    
    Parameters:
    -----------
    sigma1 : np.ndarray
        Principal stress 1 in MPa
    sigma2 : np.ndarray
        Principal stress 2 in MPa
    
    Returns:
    --------
    np.ndarray
        Von Mises stress in MPa
    
    Example:
    --------
    >>> von_mises_stress(np.array([100, 200]), np.array([50, 100]))
    array([ 86.60254038, 173.20508076])
    """
    # TODO: Implement using NumPy ufuncs
    pass


def projectile_trajectory(v0, angles, t):
    """
    Calculate projectile x, y coordinates for multiple angles.
    
    Equations:
    x = v0 * cos(theta) * t
    y = v0 * sin(theta) * t - 0.5 * g * t^2
    
    Parameters:
    -----------
    v0 : float
        Initial velocity in m/s
    angles : np.ndarray
        Launch angles in degrees
    t : np.ndarray
        Time array in seconds
    
    Returns:
    --------
    tuple of np.ndarray
        (x_positions, y_positions) - 2D arrays of shape (len(angles), len(t))
    
    Example:
    --------
    >>> x, y = projectile_trajectory(50, np.array([30, 45]), np.array([0, 1, 2]))
    >>> x.shape
    (2, 3)
    """
    g = 9.81  # m/s^2
    # TODO: Implement using NumPy ufuncs
    # Hint: Convert angles to radians, use broadcasting
    pass


def force_resultant(fx, fy):
    """
    Calculate magnitude and direction of resultant forces.
    
    Parameters:
    -----------
    fx : np.ndarray
        Force components in x-direction (N)
    fy : np.ndarray
        Force components in y-direction (N)
    
    Returns:
    --------
    tuple of np.ndarray
        (magnitude, angle_degrees) where angle is from positive x-axis
    
    Example:
    --------
    >>> mag, angle = force_resultant(np.array([3, 4]), np.array([4, 3]))
    >>> mag
    array([5., 5.])
    """
    # TODO: Implement using NumPy ufuncs
    pass


def thermal_expansion(L0, alpha, delta_T):
    """
    Calculate length change due to thermal expansion.
    
    Formula: delta_L = alpha * L0 * delta_T
    
    Parameters:
    -----------
    L0 : np.ndarray
        Original lengths in mm
    alpha : np.ndarray
        Coefficient of thermal expansion in 1/C
    delta_T : np.ndarray
        Temperature change in C
    
    Returns:
    --------
    tuple of np.ndarray
        (delta_L, L_final) - change in length and final length in mm
    
    Example:
    --------
    >>> dL, Lf = thermal_expansion(np.array([1000]), np.array([12e-6]), np.array([100]))
    >>> dL
    array([1.2])
    """
    # TODO: Implement using NumPy ufuncs
    pass


def angular_velocity_conversion(rpm):
    """
    Convert RPM to rad/s and calculate angular displacement after 5 seconds.
    
    Conversions:
    omega (rad/s) = RPM * (2*pi / 60)
    theta = omega * t
    
    Parameters:
    -----------
    rpm : np.ndarray
        Rotational speeds in RPM
    
    Returns:
    --------
    tuple of np.ndarray
        (omega, theta) - angular velocity (rad/s) and displacement (rad) after 5s
    
    Example:
    --------
    >>> omega, theta = angular_velocity_conversion(np.array([60, 120]))
    >>> omega
    array([ 6.28318531, 12.56637061])
    """
    t = 5  # seconds
    # TODO: Implement using NumPy ufuncs
    pass


def beam_deflection(x, L, w, E, I):
    """
    Calculate deflection of simply supported beam with uniform load.
    
    Formula: y = (w/(24*E*I)) * x * (L^3 - 2*L*x^2 + x^3)
    
    Parameters:
    -----------
    x : np.ndarray
        Positions along beam in m
    L : float
        Beam length in m
    w : float
        Uniform load in N/m
    E : float
        Young's modulus in Pa
    I : float
        Moment of inertia in m^4
    
    Returns:
    --------
    np.ndarray
        Deflection at each position in m (negative is downward)
    
    Example:
    --------
    >>> y = beam_deflection(np.array([0, 2.5, 5]), 5, 1000, 200e9, 1e-6)
    >>> y[0]
    0.0
    """
    # TODO: Implement using NumPy ufuncs
    pass


def velocity_components(velocities, angles):
    """
    Resolve velocities into x and y components.
    
    Parameters:
    -----------
    velocities : np.ndarray
        Velocity magnitudes in m/s
    angles : np.ndarray
        Angles from horizontal in degrees
    
    Returns:
    --------
    tuple of np.ndarray
        (vx, vy) - velocity components in m/s
    
    Example:
    --------
    >>> vx, vy = velocity_components(np.array([10, 20]), np.array([30, 45]))
    >>> vx
    array([ 8.66025404, 14.14213562])
    """
    # TODO: Implement using NumPy ufuncs
    pass


def power_to_torque(power, omega):
    """
    Calculate torque from power and angular velocity.
    
    Formula: tau = P / omega
    
    Parameters:
    -----------
    power : np.ndarray
        Power in Watts
    omega : np.ndarray
        Angular velocity in rad/s
    
    Returns:
    --------
    np.ndarray
        Torque in N*m
    
    Example:
    --------
    >>> torque = power_to_torque(np.array([1000, 2000]), np.array([100, 100]))
    >>> torque
    array([10., 20.])
    """
    # TODO: Implement using NumPy ufuncs
    pass


def spring_system(k, x):
    """
    Calculate spring force and potential energy.
    
    Formulas:
    F = -k * x
    U = 0.5 * k * x^2
    
    Parameters:
    -----------
    k : np.ndarray
        Spring constants in N/m
    x : np.ndarray
        Displacements in m
    
    Returns:
    --------
    tuple of np.ndarray
        (force, potential_energy) in N and Joules
    
    Example:
    --------
    >>> F, U = spring_system(np.array([1000, 2000]), np.array([0.1, 0.2]))
    >>> F
    array([-100., -400.])
    """
    # TODO: Implement using NumPy ufuncs
    pass


def damped_oscillation(A, b, omega, t):
    """
    Calculate displacement for damped harmonic motion.
    
    Formula: x(t) = A * exp(-b*t) * cos(omega*t)
    
    Parameters:
    -----------
    A : float
        Initial amplitude in m
    b : float
        Damping coefficient in 1/s
    omega : float
        Angular frequency in rad/s
    t : np.ndarray
        Time array in seconds
    
    Returns:
    --------
    np.ndarray
        Displacement at each time in m
    
    Example:
    --------
    >>> x = damped_oscillation(1.0, 0.1, 2*np.pi, np.array([0, 0.5, 1.0]))
    >>> x[0]
    1.0
    """
    # TODO: Implement using NumPy ufuncs
    pass