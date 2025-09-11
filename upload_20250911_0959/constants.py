"""
EBM Model Constants - IMMUTABLE PARAMETERS
==========================================

This module contains all the fixed parameters for the Energy Balance Model.
These values are defined as immutable constants and should NOT be modified.

All parameters are marked with typing.Final to prevent reassignment.
"""

from decimal import Decimal
from typing import Final

# IMMUTABLE 9-ZONE CLIMATE DATA
# These parameters are fixed and cannot be changed as per user requirements

# Climate zones (latitude ranges)
ZONES: Final[list[str]] = ["90-80", "80-70", "70-60", "60-50", "50-40", "40-30", "30-20", "20-10", "10-0"]

# Solar weighting factors for each zone
SUNWT: Final[tuple[Decimal, ...]] = (
    Decimal("0.5"), Decimal("0.531"), Decimal("0.624"), 
    Decimal("0.77"), Decimal("0.892"), Decimal("1.021"), 
    Decimal("1.12"), Decimal("1.189"), Decimal("1.219")
)

# Initial temperatures for each zone (°C)
INIT_T: Final[tuple[Decimal, ...]] = (
    Decimal("-15"), Decimal("-15"), Decimal("-5"), 
    Decimal("5"), Decimal("10"), Decimal("15"), 
    Decimal("18"), Decimal("22"), Decimal("24")
)

# Latitude centers for each zone (degrees)
LAT: Final[tuple[Decimal, ...]] = (
    Decimal("85"), Decimal("75"), Decimal("65"), 
    Decimal("55"), Decimal("45"), Decimal("35"), 
    Decimal("25"), Decimal("15"), Decimal("6")
)

# Critical temperature for ice albedo calculation (°C)
TCRIT: Final[Decimal] = Decimal("-10")

# Ice albedo value (dimensionless)
AICE: Final[Decimal] = Decimal("0.6")

# Number of zones (derived constant)
N_ZONES: Final[int] = len(ZONES)

def get_init_a(surface_albedo: Decimal) -> tuple[Decimal, ...]:
    """
    Calculate initial albedo values for each zone based on temperature.
    
    Parameters:
    -----------
    surface_albedo : Decimal
        Surface albedo value for non-ice conditions
        
    Returns:
    --------
    tuple[Decimal, ...]
        Initial albedo values for each zone
    """
    return tuple(AICE if t < TCRIT else surface_albedo for t in INIT_T)

def validate_constants() -> bool:
    """
    Validate that all constants have the correct dimensions.
    
    Returns:
    --------
    bool
        True if all constants are valid, False otherwise
    """
    expected_length = N_ZONES
    
    if len(SUNWT) != expected_length:
        return False
    if len(INIT_T) != expected_length:
        return False
    if len(LAT) != expected_length:
        return False
    
    return True

# Validate constants on import
if not validate_constants():
    raise ValueError("Constants validation failed: All arrays must have the same length")

# Prevent module-level modifications
__all__ = [
    'ZONES', 'SUNWT', 'INIT_T', 'LAT', 'TCRIT', 'AICE', 'N_ZONES',
    'get_init_a', 'validate_constants'
]
