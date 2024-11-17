"""
Decline Curve Analysis (DCA) model implementations.
Contains various decline curve equations and their derivatives.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Union, Callable


class DeclineModels:
    """
    Static class containing implementations of various decline curve models.
    
    Includes:
    - Hyperbolic decline
    - Harmonic decline
    - Exponential decline
    - Combined decline (hybrid model)
    - Model fitting utilities
    """
    
    @staticmethod
    def hyperbolic(t, qi, di, b):
        """
        Hyperbolic decline curve model.
        
        Args:
            t (array-like): Time values
            qi (float): Initial production rate
            di (float): Initial decline rate
            b (float): Hyperbolic exponent (0 < b < 1)
            
        Returns:
            array-like: Production rates at times t
            
        Notes:
            The hyperbolic decline equation is:
            q(t) = qi / (1 + b * di * t)^(1/b)
        """
        return qi / (np.abs((1 + b * di * t))**(1/b))
    
    @staticmethod
    def harmonic(t, qi, di):
        """
        Harmonic decline curve model (special case of hyperbolic where b=1).
        
        Args:
            t (array-like): Time values
            qi (float): Initial production rate
            di (float): Initial decline rate
            
        Returns:
            array-like: Production rates at times t
            
        Notes:
            The harmonic decline equation is:
            q(t) = qi / (1 + di * t)
        """
        return qi / (1 + di * t)
    
    @staticmethod
    def exponential(t, qi, di):
        """
        Exponential decline curve model (special case of hyperbolic where b=0).
        
        Args:
            t (array-like): Time values
            qi (float): Initial production rate
            di (float): Decline rate (constant)
            
        Returns:
            array-like: Production rates at times t
            
        Notes:
            The exponential decline equation is:
            q(t) = qi * exp(-di * t)
        """
        return qi * np.exp(-di * t)
    
    @staticmethod
    def hyperbolic_derivative(t, qi, di, b):
        """
        First derivative of the hyperbolic decline curve.
        Used for calculating instantaneous decline rates.
        
        Args:
            t (array-like): Time values
            qi (float): Initial production rate
            di (float): Initial decline rate
            b (float): Hyperbolic exponent
            
        Returns:
            array-like: Rate of change of production at times t
        """
        return -(qi * di * (1 + b * di * t)**(-1 - 1/b))
    
    @staticmethod
    def harmonic_derivative(t, qi, di):
        """
        First derivative of the harmonic decline curve.
        
        Args:
            t (array-like): Time values
            qi (float): Initial production rate
            di (float): Initial decline rate
            
        Returns:
            array-like: Rate of change of production at times t
        """
        return -(qi * di) / (1 + di * t)**2
    
    @staticmethod
    def exponential_derivative(t, qi, di):
        """
        First derivative of the exponential decline curve.
        
        Args:
            t (array-like): Time values
            qi (float): Initial production rate
            di (float): Decline rate
            
        Returns:
            array-like: Rate of change of production at times t
        """
        return -qi * di * np.exp(-di * t)
    
    @staticmethod
    def combined_decline_c1(t, qi, di, b, switch_time):
        """
        Combined decline curve model (hyperbolic transitioning to exponential).
        
        Args:
            t (array-like): Time values
            qi (float): Initial production rate
            di (float): Initial decline rate
            b (float): Hyperbolic exponent
            switch_time (float): Time to switch from hyperbolic to exponential
            
        Returns:
            array-like: Production rates at times t
            
        Notes:
            - Uses hyperbolic decline until switch_time
            - Transitions to exponential decline after switch_time
            - Maintains continuity of rate and decline rate at switch point
        """
        # Calculate hyperbolic decline
        hyp_decline = DeclineModels.hyperbolic(t, qi, di, b)
        
        # Calculate rate at switch point
        exp_start_rate = np.interp(switch_time, t, hyp_decline)
        
        # Calculate decline rate at switch point
        hyp_slope_at_switch = DeclineModels.hyperbolic_derivative(switch_time, qi, di, b)
        di_exp = -hyp_slope_at_switch / exp_start_rate
        
        # Calculate exponential decline after switch
        exp_decline = DeclineModels.exponential(t - switch_time, exp_start_rate, di_exp)
        
        # Combine the curves
        return np.where(t < switch_time, hyp_decline, exp_decline)

    @classmethod
    def get_model_function(cls, model_type: str) -> Tuple[Callable, tuple]:
        """
        Get the model function and default bounds for a given model type.
        
        Args:
            model_type (str): Type of model ('hyperbolic', 'harmonic', or 'exponential')
            
        Returns:
            tuple: (model_function, default_bounds)
            
        Raises:
            ValueError: If model_type is not supported
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
        Fit a decline curve model to production data.
        
        Args:
            t (array-like): Time values
            q (array-like): Production rate values
            model_type (str): Type of model to fit ('hyperbolic', 'harmonic', or 'exponential')
            bounds (tuple): Bounds for curve fitting parameters
            **kwargs: Additional arguments to pass to curve_fit
            
        Returns:
            tuple: Best fit parameters for the specified model
            
        Raises:
            ValueError: If model_type is not supported
            RuntimeError: If curve fitting fails
        """
        try:
            model_func, default_bounds = cls.get_model_function(model_type)
            bounds = bounds or default_bounds
            
            # Ensure data is numpy array
            t = np.asarray(t)
            q = np.asarray(q)
            
            # Add default kwargs for curve_fit
            kwargs.setdefault('maxfev', 5000)
            kwargs.setdefault('method', 'trf')
            
            params, _ = curve_fit(model_func, t, q, bounds=bounds, **kwargs)
            return params
            
        except RuntimeError as e:
            raise RuntimeError(f"Failed to fit {model_type} model: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error fitting {model_type} model: {str(e)}")

    @classmethod
    def get_derivative_function(cls, model_type: str) -> Callable:
        """
        Get the derivative function for a given model type.
        
        Args:
            model_type (str): Type of model ('hyperbolic', 'harmonic', or 'exponential')
            
        Returns:
            callable: Derivative function for the specified model
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type == 'hyperbolic':
            return cls.hyperbolic_derivative
        elif model_type == 'harmonic':
            return cls.harmonic_derivative
        elif model_type == 'exponential':
            return cls.exponential_derivative
        else:
            raise ValueError(f"Unsupported model type: {model_type}")