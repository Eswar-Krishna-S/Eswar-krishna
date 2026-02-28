"""
Test suite for mechanical engineering NumPy problems
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from mechanical_problems import (
    von_mises_stress,
    projectile_trajectory,
    force_resultant,
    thermal_expansion,
    angular_velocity_conversion,
    beam_deflection,
    velocity_components,
    power_to_torque,
    spring_system,
    damped_oscillation
)


class TestQ1VonMisesStress:
    """Test Q1: Von Mises Stress"""
    
    def test_single_values(self):
        result = von_mises_stress(np.array([100.0]), np.array([50.0]))
        expected = np.array([86.60254037844387])
        assert_array_almost_equal(result, expected, decimal=5)
    
    def test_array_values(self):
        sigma1 = np.array([100, 200, 150])
        sigma2 = np.array([50, 100, 75])
        result = von_mises_stress(sigma1, sigma2)
        expected = np.array([86.60254038, 173.20508076, 129.90381057])
        assert_array_almost_equal(result, expected, decimal=5)
    
    def test_zero_stress(self):
        result = von_mises_stress(np.array([0.0]), np.array([0.0]))
        assert_array_almost_equal(result, np.array([0.0]), decimal=5)


class TestQ2ProjectileTrajectory:
    """Test Q2: Projectile Motion"""
    
    def test_horizontal_launch(self):
        x, y = projectile_trajectory(10, np.array([0.0]), np.array([0, 1, 2]))
        expected_x = np.array([[0, 10, 20]])
        expected_y = np.array([[0, -4.905, -19.62]])
        assert_array_almost_equal(x, expected_x, decimal=2)
        assert_array_almost_equal(y, expected_y, decimal=2)
    
    def test_45_degree_launch(self):
        x, y = projectile_trajectory(20, np.array([45.0]), np.array([0, 1]))
        expected_x = np.array([[0, 14.142]])
        expected_y = np.array([[0, 9.237]])
        assert_array_almost_equal(x, expected_x, decimal=2)
        assert_array_almost_equal(y, expected_y, decimal=2)


class TestQ3ForceResultant:
    """Test Q3: Force Resultant"""
    
    def test_perpendicular_forces(self):
        mag, angle = force_resultant(np.array([3.0]), np.array([4.0]))
        assert_array_almost_equal(mag, np.array([5.0]), decimal=5)
        assert_array_almost_equal(angle, np.array([53.13010235]), decimal=5)
    
    def test_zero_force(self):
        mag, angle = force_resultant(np.array([0.0]), np.array([0.0]))
        assert_array_almost_equal(mag, np.array([0.0]), decimal=5)


class TestQ4ThermalExpansion:
    """Test Q4: Thermal Expansion"""
    
    def test_steel_expansion(self):
        L0 = np.array([1000.0])
        alpha = np.array([12e-6])
        delta_T = np.array([100.0])
        dL, Lf = thermal_expansion(L0, alpha, delta_T)
        assert_array_almost_equal(dL, np.array([1.2]), decimal=5)
        assert_array_almost_equal(Lf, np.array([1001.2]), decimal=5)
    
    def test_no_temperature_change(self):
        dL, Lf = thermal_expansion(np.array([1000.0]), np.array([12e-6]), np.array([0.0]))
        assert_array_almost_equal(dL, np.array([0.0]), decimal=5)


class TestQ5AngularVelocity:
    """Test Q5: Angular Velocity Conversion"""
    
    def test_60_rpm(self):
        omega, theta = angular_velocity_conversion(np.array([60.0]))
        expected_omega = np.array([2 * np.pi])
        expected_theta = np.array([10 * np.pi])
        assert_array_almost_equal(omega, expected_omega, decimal=5)
        assert_array_almost_equal(theta, expected_theta, decimal=5)


class TestQ6BeamDeflection:
    """Test Q6: Beam Deflection"""
    
    def test_end_deflection(self):
        x = np.array([0.0, 5.0])
        result = beam_deflection(x, 5.0, 1000, 200e9, 1e-6)
        assert_array_almost_equal(result, np.array([0.0, 0.0]), decimal=8)


class TestQ7VelocityComponents:
    """Test Q7: Velocity Components"""
    
    def test_horizontal_velocity(self):
        vx, vy = velocity_components(np.array([10.0]), np.array([0.0]))
        assert_array_almost_equal(vx, np.array([10.0]), decimal=5)
        assert_array_almost_equal(vy, np.array([0.0]), decimal=5)


class TestQ8PowerTorque:
    """Test Q8: Power and Torque"""
    
    def test_simple_conversion(self):
        torque = power_to_torque(np.array([1000.0]), np.array([100.0]))
        assert_array_almost_equal(torque, np.array([10.0]), decimal=5)


class TestQ9SpringSystem:
    """Test Q9: Spring Force and Energy"""
    
    def test_simple_spring(self):
        F, U = spring_system(np.array([1000.0]), np.array([0.1]))
        assert_array_almost_equal(F, np.array([-100.0]), decimal=5)
        assert_array_almost_equal(U, np.array([5.0]), decimal=5)


class TestQ10DampedOscillation:
    """Test Q10: Damped Oscillation"""
    
    def test_initial_displacement(self):
        x = damped_oscillation(1.0, 0.1, 2*np.pi, np.array([0.0]))
        assert_array_almost_equal(x, np.array([1.0]), decimal=5)