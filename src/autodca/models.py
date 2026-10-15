"""
Decline Curve Analysis (DCA) Models Module

This module implements various decline curve models used in oil and gas production analysis.
It provides mathematical models for:
1. Hyperbolic decline (most general form)
2. Harmonic decline (special case where b=1)
3. Exponential decline (special case where b=0)
4. Combined decline (hybrid model for terminal decline)

Key Features:
- Standard decline curve equations
- First derivatives for decline rate calculations
- Model fitting utilities
- Hybrid model for terminal decline behavior

Mathematical Background:
- Hyperbolic: q(t) = qi / (1 + b * di * t)^(1/b)
- Harmonic: q(t) = qi / (1 + di * t)
- Exponential: q(t) = qi * exp(-di * t)

Where:
- q(t) = production rate at time t
- qi = initial production rate
- di = initial decline rate
- b = hyperbolic exponent (0 ≤ b ≤ 1)
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Union, Callable


class DeclineModels:
    """
    Implementation of decline curve models for production forecasting.
    
    This class provides:
    1. Basic decline curve models (hyperbolic, harmonic, exponential)
    2. First derivatives for decline rate calculations
    3. Combined/hybrid models for terminal decline
    4. Utility methods for model fitting and analysis
    
    All methods are implemented as static methods for easy access without instantiation.
    """
    
    @staticmethod
    def hyperbolic(t, qi, di, b):
        """
        Hyperbolic decline curve model - most general form of decline.

        Args:
            t (array-like): Time values in years from start of analysis
            qi (float): Initial production rate
            di (float): Initial decline rate (nominal annual decline rate)
            b (float): Hyperbolic exponent (0 < b < 1)
                      b=0 reduces to exponential
                      b=1 reduces to harmonic

        Returns:
            array-like: Production rates at specified times

        Notes:
            - Most common model for unconventional wells
            - b typically ranges from 0.3 to 1.5
            - Higher b values indicate slower decline
            - Requires careful terminal decline consideration
        """
        # Handle edge case where b is very close to 0 (use exponential instead)
        if np.abs(b) < 1e-6:
            return qi * np.exp(-di * t)
        return qi / (np.abs((1 + b * di * t))**(1/b))
    
    @staticmethod
    def harmonic(t, qi, di):
        """
        Harmonic decline curve model (hyperbolic with b=1).
        
        Args:
            t (array-like): Time values in years
            qi (float): Initial production rate
            di (float): Initial decline rate
            
        Returns:
            array-like: Production rates at specified times
            
        Notes:
            - Special case of hyperbolic decline
            - Most conservative decline model
            - May overestimate reserves in many cases
            - Often seen in gas wells with significant pressure support
        """
        return qi / (1 + di * t)
    
    @staticmethod
    def exponential(t, qi, di):
        """
        Exponential decline curve model (hyperbolic with b=0).
        
        Args:
            t (array-like): Time values in years
            qi (float): Initial production rate
            di (float): Constant decline rate
            
        Returns:
            array-like: Production rates at specified times
            
        Notes:
            - Constant percentage decline rate
            - Most conservative decline model
            - Common in conventional wells
            - Semi-log plot appears as straight line
        """
        return qi * np.exp(-di * t)
    
    @staticmethod
    def hyperbolic_derivative(t, qi, di, b):
        """
        First derivative of hyperbolic decline curve.

        Args:
            t (array-like): Time values in years
            qi (float): Initial production rate
            di (float): Initial decline rate
            b (float): Hyperbolic exponent

        Returns:
            array-like: Instantaneous rate of change of production

        Notes:
            - Used for calculating instantaneous decline rates
            - Important for decline rate transitions
            - Helps identify changes in decline behavior
        """
        # Handle edge case where b is very close to 0 (use exponential derivative)
        if np.abs(b) < 1e-6:
            return -qi * di * np.exp(-di * t)
        return -(qi * di * (1 + b * di * t)**(-1 - 1/b))
    
    @staticmethod
    def harmonic_derivative(t, qi, di):
        """
        First derivative of harmonic decline curve.
        
        Args:
            t (array-like): Time values in years
            qi (float): Initial production rate
            di (float): Initial decline rate
            
        Returns:
            array-like: Rate of change of production
            
        Notes:
            - Special case of hyperbolic derivative (b=1)
            - Used for decline rate calculations
            - Important for transition analysis
        """
        return -(qi * di) / (1 + di * t)**2
    
    @staticmethod
    def exponential_derivative(t, qi, di):
        """
        First derivative of exponential decline curve.
        
        Args:
            t (array-like): Time values in years
            qi (float): Initial production rate
            di (float): Decline rate
            
        Returns:
            array-like: Rate of change of production
            
        Notes:
            - Constant percentage decline
            - Simplest derivative form
            - Used for constant terminal decline
        """
        return -qi * di * np.exp(-di * t)
    
    @staticmethod
    def combined_decline_c1(t, qi, di, b, switch_time):
        """
        Combined decline model - transitions from hyperbolic to exponential.

        Args:
            t (array-like): Time values in years
            qi (float): Initial production rate
            di (float): Initial decline rate
            b (float): Hyperbolic exponent
            switch_time (float): Time to switch from hyperbolic to exponential

        Returns:
            array-like: Production rates incorporating regime change

        Notes:
            - Models transition to terminal decline
            - Maintains continuity at switch point
            - Common in unconventional wells
            - Prevents unrealistic long-term forecasts
        """
        # Calculate initial hyperbolic decline
        hyp_decline = DeclineModels.hyperbolic(t, qi, di, b)

        # Get rate at switch point for continuity
        exp_start_rate = np.interp(switch_time, t, hyp_decline)

        # Guard against division by zero
        if exp_start_rate < 1e-10:
            return hyp_decline  # Return hyperbolic if rate is essentially zero

        # Calculate matching exponential decline rate
        hyp_slope_at_switch = DeclineModels.hyperbolic_derivative(switch_time, qi, di, b)
        di_exp = -hyp_slope_at_switch / exp_start_rate  # Terminal decline rate

        # Calculate exponential segment
        exp_decline = DeclineModels.exponential(t - switch_time, exp_start_rate, di_exp)

        # Combine curves at switch point
        return np.where(t < switch_time, hyp_decline, exp_decline)

    @classmethod
    def get_model_function(cls, model_type: str) -> Tuple[Callable, tuple]:
        """
        Get model function and appropriate parameter bounds.
        
        Args:
            model_type: Name of decline model ('hyperbolic', 'harmonic', 'exponential')
            
        Returns:
            tuple: (model_function, parameter_bounds)
            
        Raises:
            ValueError: For unsupported model types
            
        Notes:
            Default bounds ensure physically meaningful parameters:
            - Production rates (qi) ≥ 0
            - Decline rates (di) ≥ 0
            - Hyperbolic exponent (b) between 0 and 1
        """
        if model_type == 'hyperbolic':
            return cls.hyperbolic, ((0, 0, 0), (np.inf, np.inf, 1))
        elif model_type == 'harmonic':
            return cls.harmonic, ((0, 0), (np.inf, np.inf))
        elif model_type == 'exponential':
            return cls.exponential, ((0, 0), (np.inf, np.inf))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @classmethod
    def fit_model(cls, t, q, model_type='hyperbolic', bounds=None, **kwargs):
        """
        Fit decline curve model to production data.
        
        Args:
            t: Time values in years
            q: Production rate values
            model_type: Type of model to fit
            bounds: Optional parameter bounds
            **kwargs: Additional arguments for scipy.optimize.curve_fit
            
        Returns:
            tuple: Best fit parameters for specified model
            
        Raises:
            ValueError: For invalid inputs or fitting failures
            RuntimeError: If curve fitting fails to converge
            
        Notes:
            - Uses scipy's curve_fit with robust defaults
            - Handles common fitting issues
            - Returns optimal parameters for forecasting
            - For hyperbolic decline, b parameter is rounded to nearest 0.01
            - When rounded b ≈ 0, treats as exponential decline
            - When rounded b ≈ 1, treats as harmonic decline
        """
        try:
            # Get model function and default bounds
            model_func, default_bounds = cls.get_model_function(model_type)
            bounds = bounds or default_bounds
            
            # Ensure numpy arrays
            t = np.asarray(t)
            q = np.asarray(q)
            
            # Set robust default fitting parameters
            kwargs.setdefault('maxfev', 5000)  # Maximum function evaluations
            kwargs.setdefault('method', 'trf')  # Trust Region Reflective algorithm
            
            # Perform the fit
            params, _ = curve_fit(model_func, t, q, bounds=bounds, **kwargs)
            
            # Handle b parameter rounding for hyperbolic model
            if model_type == 'hyperbolic':
                qi, di, b = params
                b_rounded = round(b, 2)  # Round to nearest 0.01
                
                # If b is very close to 0 (rounded to 0), treat as exponential
                if b_rounded == 0:
                    return (qi, di, 0.0)  # Force exactly 0 for exponential
                # If b is very close to 1 (rounded to 1), treat as harmonic
                elif b_rounded == 1:
                    return (qi, di, 1.0)  # Force exactly 1 for harmonic
                else:
                    return (qi, di, b_rounded)
            
            return params
            
        except RuntimeError as e:
            raise RuntimeError(f"Failed to fit {model_type} model: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error fitting {model_type} model: {str(e)}")

    @classmethod
    def get_derivative_function(cls, model_type: str) -> Callable:
        """
        Get derivative function for decline rate calculations.
        
        Args:
            model_type: Name of decline model
            
        Returns:
            callable: Corresponding derivative function
            
        Raises:
            ValueError: For unsupported model types
            
        Notes:
            - Used for decline rate analysis
            - Important for transition point calculations
            - Helps validate model behavior
        """
        if model_type == 'hyperbolic':
            return cls.hyperbolic_derivative
        elif model_type == 'harmonic':
            return cls.harmonic_derivative
        elif model_type == 'exponential':
            return cls.exponential_derivative
        else:
            raise ValueError(f"Unsupported model type: {model_type}")