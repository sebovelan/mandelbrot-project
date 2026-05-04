import pytest
import numpy as np
from naive import mandelbrot_naive
from numpy_version import mandelbrot_numpy

# We will test both versions to be thorough
FUNCTIONS = [mandelbrot_naive, mandelbrot_numpy]

@pytest.mark.parametrize("func", FUNCTIONS)
def test_output_shape(func):
    """
    Test Case 1: Ensure the output shape matches the requested width and height.
    """
    width = 100
    height = 50
    max_iter = 10
    
    result = func(width, height, max_iter)
    
    assert result.shape == (height, width), f"Expected shape {(height, width)}, got {result.shape}"

@pytest.mark.parametrize("func", FUNCTIONS)
def test_inside_set(func):
    """
    Test Case 2: Ensure a point deep inside the set (like origin) reaches max_iter
    and therefore returns a value of 1.0 (since it's normalized as iteration/max_iter).
    
    The coordinate system in naive/numpy goes from x: [-2, 1] and y: [-1.5, 1.5].
    The center roughly corresponds to x=0, y=0.
    For width=300, height=300:
    x=0 is at index j = 200 (since x_min=-2, range=3, 2/3 of 300 = 200)
    y=0 is at index i = 150 (since y_min=-1.5, range=3, 1.5/3 of 300 = 150)
    """
    width = 300
    height = 300
    max_iter = 50
    
    result = func(width, height, max_iter)
    
    # Point at (0,0) shouldn't escape
    center_val = result[150, 200]
    
    # Since we normalize by dividing by max_iter, it should be 1.0 (or very close to it depending on max_iter handling)
    assert center_val == 1.0, f"Expected 1.0 for point inside set, got {center_val}"

@pytest.mark.parametrize("func", FUNCTIONS)
def test_outside_set(func):
    """
    Test Case 3: Ensure a point far outside the set escapes quickly.
    
    A point at x=-2, y=-1.5 (index 0,0) will escape quickly.
    """
    width = 100
    height = 100
    max_iter = 50
    
    result = func(width, height, max_iter)
    
    # Point at (-2, -1.5) escapes in just a couple of iterations
    # So its normalized value will be < 1.0
    corner_val = result[0, 0]
    
    assert corner_val < 1.0, f"Expected < 1.0 for point outside set, got {corner_val}"
