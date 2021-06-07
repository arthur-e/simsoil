'''
Functions related to calculating (evapo)transpiration. These are largely taken
from the MODIS MOD16 framework, but the goal is not to reproduce MOD16;
rather, they facilitate the calculation of transpiration as one of the three
components of evapotranspiration. Various primary sources are cited in the
MOD16 code:

- Monteith, J. L., and M. Unsworth. 2001. "Principles of Environmental
    Physics", Second Ed.
- Mu, Q., Zhao, M., & Running, S. W. (2011). Improvements to a MODIS
  global terrestrial evapotranspiration algorithm. *Remote Sensing of
  Environment*, 115(8), 1781–1800.

**NOTE:** Currently, canopy evaporation and potential transpiration are
denominated in seconds (i.e., kg m-2 sec-1) instead of in days, as in MOD16.
This is for compatibility with soil water infiltration models that run in
with sub-daily time steps. As such, net radiation to the land surface or
canopy should be in units of J m-2 s-1.
'''

import numpy as np

STEFAN_BOLTZMANN = 5.67e-8 # Stefan-Boltzmann constant, W m-2 K-4
SPECIFIC_HEAT_CAPACITY_AIR = 1013 # J kg-1 K-1, Monteith & Unsworth (2001)
# Ratio of molecular weight of water vapor to that of dry air (ibid.)
MOL_WEIGHT_WET_DRY_RATIO_AIR = 0.622

# Calculate the latent heat of vaporization (J kg-1)
latent_heat_vapor = lambda temp_k: (2.501 - 0.002361 * (temp_k - 273.15)) * 1e6

def canopy_evaporation(
        pressure, temp_k, rhumidity, vpd, lai, fpar, rad_canopy, f_wet = None,
        g_h = 0.01, g_e = 0.01):
    r'''
    Wet canopy evaporation calculated according to the MODIS MOD16 framework,
    with air density calculation from the [
    National Physical Laboratory (2021),
    "Buoyancy Correction and Air Density Measurement."
    ](http://resource.npl.co.uk/docs/science_technology/mass_force_pressure/
        clubs_groups/instmc_weighing_panel/buoycornote.pdf)

    $$
    \lambda E = \frac{(s\, A_c\, F_c + \rho\, C_p\, D\, (F_c / r_a))F_{wet}}{s +
        (P_a\, C_p\, r_c)(\lambda\, \varepsilon\, r_a)^{-1}}
    $$

    Parameters
    ----------
    pressure : float or numpy.ndarray
        The air pressure in Pascals
    temp_k : float or numpy.ndarray
        The air temperature in degrees K
    rhumidity : float or numpy.ndarray
        Relative humidity, as a proportion on [0,1]
    vpd : float or numpy.ndarray
        The vapor pressure deficit in Pascals
    lai : float or numpy.ndarray
        The leaf area index (LAI)
    fpar : float or numpy.ndarray
        Fraction of photosynthetically active radiation (PAR) absorbed by
        the vegetation canopy
    rad_canopy : float or numpy.ndarray
        Net radiation to the canopy (J m-2 s-1)
    f_wet : float or numpy.ndarray or None
        (Optional) Fraction of the land surface that is saturated/ covered
        with standing water; if None, calculates this fraction like MOD16,
        based on the relative humidity
    g_h : float
        Leaf conductance to sensible heat per unit LAI
        (Default: 0.01 m s-1 LAI-1)
    g_e : float
        Leaf conductance to evaporated water per unit LAI
        (Default: 0.01 m s-1 LAI-1)

    Returns
    -------
    float or numpy.ndarray
        Evaporation from the wet canopy surface (kg m-2 s-1)
    '''
    assert np.logical_and(0 <= rhumidity, rhumidity <= 1),\
        'Relative humidity (rhumidity) must be on the interval [0,1]'
    if f_wet is None:
        f_wet = np.where(rhumidity < 0.7, 0, np.power(rhumidity, 4))
    # NIST simplified air density formula with buoyancy correction (NPL 2021)
    rho = np.divide( # Convert Pa to mbar, RH to RH% (percentage)
        0.348444 * (pressure / 100) - (rhumidity * 100) *\
            (0.00252 * temp_k - 273.15 - 0.020582),
        temp_k) # kg m-3
    # Wet canopy resistance to sensible heat ("rhc")
    r_h = 1 / (g_h * lai * f_wet)
    # Wet canopy resistance to evaporated water on the surface ("rvc")
    r_c = 1 / (g_e * lai * f_wet)
    # Resistance to radiative heat transfer through air ("rrc")
    r_r = np.divide(
        rho * SPECIFIC_HEAT_CAPACITY_AIR,
        4 * STEFAN_BOLTZMANN * np.power(temp_k, 3))
    # Aerodynamic resistance to evaporated water on the wet canopy surface
    r_a = np.divide(r_h * r_r, r_h + r_r) # (s m-1)
    # Slope of saturation vapor pressure curve
    s = svp_slope(temp_k)
    # Latent heat of vaporization (J kg-1)
    lhv = latent_heat_vapor(temp_k)
    # Mu et al. (2011), Equation 17; PET (J m-2 s-1) is divided by the latent
    #   heat of vaporization (J kg-1) to obtain mass flux (kg m-2 s-1)
    return np.divide(
        f_wet * ((s * rad_canopy) +\
            (rho * SPECIFIC_HEAT_CAPACITY_AIR * vpd * (fpar * 1/r_a))),
        s + ((pressure * SPECIFIC_HEAT_CAPACITY_AIR * r_c) *\
            1/(lhv * MOL_WEIGHT_WET_DRY_RATIO_AIR * r_a))) / lhv


def psychrometric_constant(pressure, temp_k):
    r'''
    The psychrometric constant, which relates the vapor pressure to the air
    temperature. Calculation derives from the "Handbook of Hydrology" by D.R.
    Maidment, Section 4.2.28.

    $$
    \gamma = \frac{C_p \times P}{\lambda\times 0.622}
    $$

    Where `C_p` is the specific heat capacity of air, `P` is air pressure, and
    `lambda` is the latent heat of vaporization.

    Parameters
    ----------
    pressure : float or numpy.ndarray
        The air pressure in Pascals
    temp_k : float or numpy.ndarray
        The air temperature in degrees K

    Returns
    -------
    float or numpy.ndarray
        The psychrometric constant at this pressure, temperature (Pa K-1)
    '''
    lhv = latent_heat_vapor(temp_k) # Latent heat of vaporization (J kg-1)
    return (SPECIFIC_HEAT_CAPACITY_AIR * pressure) /\
        (lhv * MOL_WEIGHT_WET_DRY_RATIO_AIR)


def radiation_net(sw_rad, sw_albedo, temp_k):
    r'''
    Net incoming radiation to the land surface, calculated according to the
    MOD16 algorithm and Cleugh et al. (2007); see Equation 7 in the MODIS
    MOD16 Collection 6.1 User's Guide.

    - Cleugh, H. A., Leuning, R., Mu, Q., & Running, S. W. (2007).
      Regional evaporation estimates from flux tower and MODIS satellite data.
      *Remote Sensing of Environment*, 106(3), 285–304.

    $$
    R_{net} = (1 - \alpha)\times R_{S\downarrow} +
        (\varepsilon_a - \varepsilon_s) \times \sigma \times T^4
    \quad\mbox{where}\quad \varepsilon_s = 0.97
    $$

    Where `alpha` is the MODIS albedo, `R_S` is down-welling short-wave
    radiation, `sigma` is the Stefan-Boltzmann constant, and:

    $$
    \varepsilon_a = 1 - 0.26\,\mathrm{exp}\left(
      -7.77\times 10^{-4}\times (T - 273.15)^2
    \right)
    $$

    Parameters
    ----------
    sw_rad : float or numpy.ndarray
        Incoming short-wave radiation (W m-2)
    sw_albedo : float or numpy.ndarray
        White-sky albedo, e.g., from MODIS MCD43A3
    temp_k : float or numpy.ndarray
        Air temperature in degrees K

    Returns
    -------
    float or numpy.ndarray
        Net incoming radiation to the land surface (W m-2)
    '''
    # Mu et al. (2011), Equation 5
    emis_surface = 0.97
    emis_atmos = 1 - 0.26 * np.exp(-7.77e-4 * np.power(temp_k - 273.15, 2))
    return sw_rad * (1 - sw_albedo) +\
        STEFAN_BOLTZMANN * (emis_atmos - emis_surface) * np.power(temp_k, 4)


def svp(temp_k):
    r'''
    The saturation vapor pressure, based on [
    the Food and Agriculture Organization's (FAO) formula, Equation 13
    ](http://www.fao.org/3/X0490E/x0490e07.htm).

    $$
    \mathrm{SVP} = 1\times 10^3\left(
    0.6108\,\mathrm{exp}\left(
      \frac{17.27 (T - 273.15)}{T - 273.15 + 237.3}
      \right)
    \right)
    $$

    Parameters
    ----------
    temp_k : float or numpy.ndarray
        The air temperature in degrees K

    Returns
    -------
    float or numpy.ndarray
    '''
    temp_c = temp_k - 273.15
    # And convert from kPa to Pa
    return 1e3 * 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def svp_slope(temp_k):
    r'''
    The slope of the saturation vapour pressure curve, which describes the
    relationship between saturation vapor pressure and temperature. Based on [
    the Food and Agriculture Organization's (FAO) formula, Equation 13
    ](http://www.fao.org/3/X0490E/x0490e07.htm).

    $$
    \Delta = 4098\times [\mathrm{SVP}]\times (T - 273.15 + 237.3)^{-2}
    $$

    Parameters
    ----------
    temp_k : float or numpy.ndarray
        The air temperature in degrees K

    Returns
    -------
    float or numpy.ndarray
        The slope of the saturation vapor pressure curve in Pascals per
        degree K (Pa degK-1)
    '''
    temp_c = temp_k - 273.15
    svp = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    # With conversion of SVP from kPa to Pa
    return (4098 * (svp * 1e3)) / np.power(temp_c + 237.3, 2)


def transpiration_potential(
        air_temp_k, pressure, fpar, sw_rad, sw_albedo, rhumidity = None,
        f_wet = None):
    r'''
    Estimates potential transpiration in each soil layer. Transpiration is
    calculated using the Priestly-Taylor method for "potential transpiration,"
    as described in the MODIS MOD16 framework.

    $$
    \lambda T = \frac{\alpha\, s\, A_c\, (1 - F_{wet})}{\gamma\, s}
    $$

    Where `alpha` is the MODIS albedo, `s` is the slope of the saturation
    vapor pressure curve or `svp_slope()`, `A_c` is the net radiation
    intercepted by the canopy, `F_wet` is the fraction of the land surface
    that is saturated, and `gamma` is the `psychrometric_constant()`.

    Parameters
    ----------
    air_temp_k : numpy.ndarray
        Air temperature (deg K)
    pressure : numpy.ndarray
        Air pressure (Pa)
    fpar : numpy.ndarray
        Fraction of photosynthetically active radiation (PAR) absorbed by
        the vegetation canopy
    sw_rad : numpy.ndarray
        Incoming short-wave radiation, (W m-2) or (J s-1 m-2)
    sw_albedo : numpy.ndarray
        White-sky albedo for short-wave radiation
    rhumidity: numpy.ndarray or None
        (Optional) The relative humidity, expressed as a proportion on the
        interval [0,1]; if not provided, f_wet must be provided.
    f_wet : numpy.ndarray or None
        (Optional) Fraction of the land surface that is saturated/ covered
        with standing water; if None, calculates this fraction like MOD16,
        based on the relative humidity (rhumidity).

    Returns
    -------
    numpy.ndarray
        A (Z x 1) array of transpiration in each soil layer (kg m-2 s-1)
    '''
    assert rhumidity is not None or f_wet is not None,\
        'One must be provided: "rhumidity" or "f_wet"'
    if rhumidity is not None:
        assert np.logical_or(np.isnan(rhumidity),
                np.logical_and(0 <= rhumidity, rhumidity <= 1)).all(),\
            'Relative humidity (rhumidity) must be on the interval [0,1]'
    if f_wet is None:
        f_wet = np.where(rhumidity < 0.7, 0, np.power(rhumidity, 4))
    alpha = 1.26 # Equation 18, MOD16 Collection 6.1 User's Guide
    s = svp_slope(air_temp_k) # Slope of saturation vapor pressure curve
    gamma = psychrometric_constant(pressure, air_temp_k)
    # Net radiation to the land surface
    net_rad = radiation_net(sw_rad, sw_albedo, air_temp_k)
    # Net radiation to the canopy
    canopy_rad = np.multiply(fpar, net_rad)
    pt = np.divide(alpha * s * canopy_rad * (1 - f_wet), s + gamma)
    # Divide by latent heat of vaporization to convert to a mass flux
    return pt / latent_heat_vapor(air_temp_k)
