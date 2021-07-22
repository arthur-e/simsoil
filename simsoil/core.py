'''
Soil water infiltration model based on the modified Richards equation from
the Community Land Model (CLM), version 5.0, with some equations from CLM 4.x
and pedotransfer functions from Balland et al. (2008). See also:

- Balland, V., Pollacco, J. A. P., & Arp, P. A. (2008). Modeling soil
    hydraulic properties for a wide range of soil conditions.
    *Ecological Modelling*, 219(3–4), 300–316.
- Jackson, R. B., Canadell, J., Ehleringer, J. R., Mooney, H. A., Sala,
    O. E., & Schulze, E. D. (1996). A global analysis of root distributions
    for terrestrial biomes. *Oecologia*, 108, 389–411.
- Tolk, J. A. (2003). Soils, Permanent Wilting Points.
    *Encyclopedia of Water Science*, 927–929.
- Verhoef, A., & Egea, G. (2014). Modeling plant transpiration under limited
    soil water: Comparison of different plant and soil hydraulic
    parameterizations and preliminary implications for their use in land
    surface models. *Agricultural and Forest Meteorology*, 191, 22–32.
- Zeng, X., & Decker, M. (2008). Improving the numerical solution of soil
    moisture-based Richards equation for land models with a deep or shallow
    water table. *Journal of Hydrometeorology*, 10(1), 308–319.
'''

import numpy as np
from cached_property import cached_property
from simsoil import Namespace, suppress_warnings, tridiag_solver
from simsoil.transpiration import latent_heat_vapor, psychrometric_constant, radiation_net, svp_slope

# From the Catchment land model of SMAP L4SM
DEPTHS = -np.array((0.05, 0.15, 0.35, 0.75, 1.5, 3.0)).reshape((6,1)) # meters
# Scaling ratios for soil organic carbon (i.e., ratio of volumetric SOC
#   content, relative to top layer), from Endsley et al. (2020)
SOC_RATIOS = np.array((1, 2.35, 4.25, 6.46, 9.56, 12.77)).reshape((6,1))

class InfiltrationModel(object):
    '''
    A soil water infiltration model, based on the SoilProfile class and
    facilitating a maximum infiltration rate, transpiration loss, sub-surface
    drainage, and with adaptive time stepping. Outstanding issues:

    1. Frozen layers may exceed the saturation porosity because liqud water
        content can't be moved to or from those layers during rebalancing.

    Parameters
    ----------
    soil_model : SoilProfile
    dt_min : int
        Minimum number of seconds a sub-daily time step can take
    f_ice_cutoff : float
        Ice fraction cutoff on the interval [0, 1], but the value should
        be >/= 0.95. If the ice fraction exceeds this value, a rebalancing
        of soil moisture will not be performed. This can be necessary to
        avoid running into impossible balancing scenarios.
    debug : bool
        True to perform some (potentially expensive) validation checks at
        runtime (Default: False)
    '''
    SECS_PER_DAY = 86400 # Number of seconds per day

    def __init__(
            self, soil_model, dt_min = 10, f_ice_cutoff = 0.96,
            debug = False):
        self._debug = debug
        self._dt0 = dt_min # Minimum number of seconds for each time step
        self._f_ice_cutoff = f_ice_cutoff
        self.soil = soil_model

    def run(
            self, vwc, temp_profile, transpiration, influx, f_saturated, dt,
            n_days = None, ltol = 1e-2, utol = 1e-1, climatology = False,
            adaptive = True):
        r'''
        Runs the soil water infiltration model forward in time for a certain
        number of days.

        NOTE: If not running in debug mode (`debug = True` when initialized),
        the matric potential will be `None`.

        When `adaptive = True`, adaptive time stepping is used and the number
        of time steps, `dt` is a starting point; the sub-daily time step size
        will be adjusted at the end of each day based on the estimated
        temporal truncation error, which is calculated:

        $$
        \epsilon_i = \left(
        \frac{\Delta \theta_{liq,i}\, \Delta z_i}{\Delta t} -
        (q_{i-1} - q_i + e_i)
        \right) \frac{\Delta t}{2}
        $$

        If the error is less than `ltol`, sub-daily time steps are doubled in
        size; if the error is greater than `utol`, time steps are halved.

        Parameters
        ----------
        vwc : numpy.ndarray
            (Z x 1) array of the initital soil volumetric water content (VWC)
            profile
        temp_profile : numpy.ndarray
            (Z x 1) array of soil temperatures in degrees K for the current
            time step
        transpiration : list or tuple or numpy.ndarray or None
            Sequence of the total daily potential (unconstrained)
            transpiration rate (kg m-2 s-1)
        influx: list or tuple or numpy.ndarray
            Sequence of the daily water infiltration rate at the surface
            layer, in units of (kg m-2 s-1) or (mm s-1), as 1 mm of water over
            an area of 1 m-2 weighs 1 kg.
        f_saturated : list or tuple or numpy.ndarray
            Sequence of the daily fraction of the land surface that is
            saturated
        dt : int
            Size of time step (secs)
        n_days : int or None
            Number of days to run; defaults to the size of `influx`
        ltol : float
            Lower bound for error tolerance in adaptive time stepping
        utol : float
            Upper bound for error tolerance in adaptive time stepping
        climatology : bool
            True to run in climatology mode; i.e., if the input driver data
            are a 365-day climatology, the day index should be recycled
        adaptive : bool
            True to use adaptive time stepping: dynamic adjustment of the
            sub-daily time step based on the error in water balance
            (Default: True)

        Returns
        -------
        tuple
            3-tuple of `(vwc, err, psi)` where `vwc` is the soil moisture time
            series, a (Z x T) array; `err` is the estimated truncation error,
            a (Z x T) array; `psi` is the estimated soil matric potential, a
            (Z x T) array, where T is time and Z is the number of layers.
        '''
        if n_days is None:
            n_days = influx.size
        n_layers = self.soil._depths_m.size
        est_vwc = np.ones((n_layers, n_days)) * np.nan
        est_error = np.ones((n_layers, n_days)) * np.nan
        est_psi = None
        if self._debug:
            est_psi = np.ones((n_layers, n_days)) * np.nan
            if not climatology:
                assert len(transpiration) == n_days
        # If the data are a 365-day climatology, we must recycle the day
        #   index; d % 365 is on closed interval [0,364]
        iterations = np.arange(0, n_days)
        if climatology:
            iterations = iterations % 365
        for i, d in enumerate(iterations):
            successful = False
            while not successful and (self._dt0 <= dt < self.SECS_PER_DAY // 2):
                # try:
                args = [
                    vwc, temp_profile[:,d,None],
                    transpiration[d] if transpiration is not None else None,
                    influx[d], f_saturated[d], dt
                ]
                if self._debug:
                    vwc, de = self.step_daily(*args)
                else:
                    try:
                        vwc, de = self.step_daily(*args)
                    except:
                        print('ERROR: Ending prematurely due to error in InfiltrationModel.step_daily()')
                        return (est_vwc, est_error, est_psi)
                err = np.abs(np.stack(de)).mean(axis = 0).max()
                if not adaptive or ltol < err <= utol:
                    successful = True
                    continue # Accept the result if within error bounds
                if err <= ltol:
                    # Accept the result if error is lower than expected,
                    #   but double the size of the time stpes
                    successful = True
                # Double step size if error too small; halve if too large
                d_dt = 2 if err <= ltol else 0.5
                if self._dt0 <= (dt * d_dt) < self.SECS_PER_DAY // 2:
                    dt = int(dt * d_dt)
            # Take mean over all steps of the maximum absolute error at any depth
            est_vwc[:,i] = vwc.ravel()
            est_error[:,i] = np.stack(de).mean(axis = 0).ravel()
            if self._debug:
                f_ice = self.soil.f_ice(vwc, temp_profile[:,d,None])
                psi = self.soil.matric_potential(vwc, f_ice)
                est_psi[:,i] = psi.ravel()
        return (est_vwc, est_error, est_psi)

    def step_daily(
            self, vwc, temp_profile, transpiration, influx, f_saturated, dt):
        '''
        Executes a single daily time step of the soil water infiltration
        model. There are five distinct steps: 1) The maximum soil water
        infiltration rate is calculated and excess influx is removed; 2) The
        potential transpiration is converted to actual transpiration based on
        the current soil moisture stress; 3) The updated soil VWC in each
        layer is solved for; 4) Drainage from the bottom layer is removed from
        the updated soil VWC profile; 5) Water contents of each layer are
        checked for saturation to ensure they are within reasonable bounds.

        The last step ("rebalancing") is potentially the most intensive and
        needs to be re-implemented, probably in Cython. Currently, we cannot
        guarantee that the bottom-most layer doesn't drain completely in very
        dry conditions. Frozen soils are also tricky; if the maximum liquid
        water content the soil can hold is below 1 mm, we do not attempt to
        rebalance liquid water.

        See `SoilProfile.solve_vwc()` for how sub-surface runoff and specific
        yield are calculated.

        Parameters
        ----------
        vwc : numpy.ndarray
            (Z x 1) array of the initital soil volumetric water content (VWC)
            profile
        temp_profile : numpy.ndarray
            (Z x 1) array of soil temperatures in degrees K for the current
            time step
        transpiration : float
            Total potential (unconstrained) transpiration rate (kg m-2 s-1),
            a daily scalar value
        influx: float
            Scalar or N-dimensional array of water infiltration at the surface
            layer, in units of (kg m-2 s-1) or (mm s-1), as 1 mm of water
            over an area of 1 m-2 weighs 1 kg.
        f_saturated : float
            The fraction of the land surface that is saturated
        dt : int
            Size of time step (secs)

        Returns
        -------
        tuple
            2-tuple of `(vwc, error)` where `vwc` is the updated soil moisture
            profile and `error` is the estimated truncation error.
        '''
        def rebalance(vwc, temp_k, thickness_mm):
            # Rebalance water content of all soil layers
            # To speed things up, if temps. well above freezing, f_ice = 0
            if (temp_k > 276).all():
                f_ice = np.zeros((vwc.shape))
            else:
                f_ice = self.soil.f_ice(vwc, temp_k)
            # Calculate liquid and ice water contents
            wliq = (vwc - (vwc * f_ice)) * -thickness_mm
            # wice = (vwc * f_ice) / (-thickness_mm * (self.soil.DENSITY_ICE / 1e3))
            wliq_max = (self.soil._theta_sat - (vwc * f_ice)) * -thickness_mm
            i = 0
            while not np.logical_or(
                    np.logical_and(0.01 <= wliq, wliq <= wliq_max),
                    wliq_max < 0.01 # e.g., for layers with f_ice --> 1.0
                ).all():
                assert i < 1000 # Guard against non-convergence
                excess = np.where(wliq > wliq_max, wliq - wliq_max, 0)
                deficit = np.where(wliq < 0.01, 0.01 - wliq, 0)
                # If a layer is ice-packed/ can't hold water don't move any
                #   water to or from this layer
                excess = np.where(wliq_max < 0.01, 0, excess)
                deficit = np.where(wliq_max < 0.01, 0, deficit)
                # Excess is moved to layer above (for surface layer, it is
                #   discarded, i.e., as runoff); deficit moved from below
                wliq -= excess + np.vstack((excess[1:], 0))
                wliq += deficit - np.vstack((0, deficit[:-1]))
                i += 1
            else:
                # Add the liquid and ice contents back together; (vwc * f_ice)
                #   is already in volumetric terms
                vwc = (wliq / -thickness_mm) + (vwc * f_ice)
            return vwc

        de = [] # Temporal truncation error estimates
        iterations = range(0, 86400 // dt)
        thickness_mm = self.soil._thickness_mm
        if self._debug:
            assert not hasattr(transpiration, '__len__'),\
                'Transpiration should be a scalar value (float)'
        for t in iterations:
            # Attempt to solve system of equations; VWC should be on [0,1]
            assert np.logical_and(0 <= vwc, vwc <= 1).all(),\
                'VWC out of bounds'
            actual_trans = np.zeros(vwc.shape)
            if transpiration is not None:
                actual_trans, _, _ = self.soil.solve_sink(vwc, transpiration)
            # Calculate maximum infiltration rate
            max_influx = self.soil.max_infiltration(
                vwc, temp_profile, f_saturated)
            # Then, solve for change in VWC
            #   t+1 estimate of d(VWC), t estimates of q_in and q_out;
            #   implicitly remove surface runoff, by max. infiltration rate
            x, flows, runoff = self.soil.solve_vwc(
                min(influx, max_influx[0]), vwc, temp_profile, dt,
                actual_trans)
            q_in, q_out = flows
            # Update VWC, then subtract lateral sub-surface runoff
            vwc = np.add(vwc, x)
            vwc = vwc + runoff # Runoff is negative
            # Finally, check that water contents of each layer are within
            #   bounds (i.e., not saturated)
            vwc = rebalance(vwc, temp_profile, thickness_mm)
            # Estimate truncation error (ignoring drainage because it is not
            #   considered in the tridiagonal equation), update results
            if t > 0:
                err = (dt / 2) * (
                    ((x * thickness_mm) / dt) -\
                    (q_in0 - q_out0 - actual_trans))
            q_in0, q_out0 = (q_in, q_out)
            if t > 0:
                de.append(err)
        return (vwc, de)


class SoilProfile(object):
    '''
    Represents a soil profile. In this first version, the total porosity and
    the sand and clay fractions are scalar fields that represent the entire
    soil column. Total porosity stands in for the saturation porosity, which
    is a function of organic matter and sand content in CLM 4.0. These scalar
    fields are also fixed throughout the simulation; i.e., changes in soil
    organic carbon as part of some coupled soil decomposition model do not
    propagate to changes in the organic fraction.

    Everything except the solution to the tridiagonal equation is vectorized
    so, for now, input arrays must be (Z x N) for N = 1 only. As part of this
    limitation, there is a check that the bedrock depth is a scalar.

    NOTES:

    1. Instead of tracking ice and liquid water content separately, VWC
        generally refers to the total (ice plus liquid) volumetric water
        content. The liquid water content is determined based on the ice
        fraction, when needed.
    2. Hydraulic condutivity (k) is always measured at the bottom interface of
        a soil layer.

    Parameters
    ----------
    pft : int
        Plant functional type (PFT) code
    soc : numpy.ndarray
        Areal soil organic carbon (SOC) content (g C m-2) in each layer;
        should be a (Z x 1) array
    sand : float or numpy.ndarray
        Sand content of soil, as proportion on [0,1], a (Z x 1) array
    clay : float or numpy.ndarray
        Clay content of soil, as proportion on [0,1], a (Z x 1) array
    porosity : float or numpy.ndarray
        Total porosity of the soil; should be a scalar or a (Z x 1) array
        on [0,1]
    bedrock : float
        Depth to bedrock (m); not currently used, so any numeric value can be
        provided
    slope : float
        Topographic slope (degrees)
    depths_m : numpy.ndarray
        Depths of each soil layer's bottom interface, in meters; should be
        a (Z x 1) array with negative values below the surface
    '''
    DENSITY_ICE = 917 # kg m-3 (Cutnell & Johnson. 1995. "Physics," 3rd ed.)
    DENSITY_WATER = 1000 # kg m-3
    DRAINAGE_DECAY_FACTOR = 2.5 # m-1 (CLM 4.5)
    MATRIC_POTENTIAL_ORGANIC = -10.3 # mm (CLM 4.0)
    PERCOLATION_THRESHOLD = 0.5 # From percolation theory and CLM 4.0
    SOCC_MAX = 130e3 # g m-3 (Lawrence and Slater 2008)
    SOIL_FREEZING = 273.15 # Temp. below which soil is frozen (deg K)

    def __init__(
            self, pft, soc, sand, clay, porosity, bedrock, slope,
            depths_m = DEPTHS, debug = False):
        assert soc.ndim == 2 and depths_m.ndim == 2,\
            'SOC and depths arrays must be 2-dimensional'
        assert soc.shape[0] == depths_m.size,\
            'Need one SOC value per soil layer'
        assert np.all(depths_m < 0),\
            'Depths should be defined as negative downward from the soil surface'
        assert not hasattr(bedrock, '__len__'),\
            'Only one bedrock depth should be provided'
        self._depths_m = depths_m
        self._frac_clay = clay
        # Convert from areal to volumetric SOCC, then to fraction
        self._frac_organic = (soc / np.abs(self._depths_m)) / self.SOCC_MAX
        assert np.nanmax(self._frac_organic) < 1,\
            'Organic fraction > 1.0; check units of soil organic carbon'
        self._frac_sand = sand
        self._pft = pft
        self._porosity = porosity
        self._slope = np.deg2rad(slope)
        self._thickness_m = (self._depths_m - np.vstack((0, self._depths_m[:-1])))\
            .astype(np.float32)
        self._thickness_mm = (self._thickness_m * 1e3).astype(np.float32)
        # Depth to bottom interface (mm)
        self._z = -np.abs(depths_m) * 1e3
        # Depth to the bedrock (mm)
        self._z_bedrock = -np.abs(bedrock) * 1e3
        # Depth to "node" (i.e., middle of the soil layer)
        self._z_node = (self._depths_m * 1e3) - self._thickness_mm / 2
        # Define namespace for free parameters
        self._zeros = np.zeros(self._depths_m.shape)
        self.params = Namespace()
        self.params.add('ksat_om', 1e-1) # mm s-1
        self.params.add('alpha', 3) # CLM 4.0 Tech Note, Section 7.3.0

    @cached_property
    def _b(self):
        'The Clapp & Hornberger exponent, "B"'
        # Equation 7.84 in CLM 4.0 Tech Note 7.4.1
        b_min = 2.91 + 0.159 * (self._frac_clay * 100)
        # Where B_{om} = 2.7; see Letts et al. 2000
        return ((1 - self._frac_organic) * b_min) + (self._frac_organic * 2.7)

    @cached_property
    @suppress_warnings
    def _frac_percolating(self):
        'Fraction of soil layer allows percolation in organic material'
        # Equation 7.88 in CLM 4.0 Tech Note, Section 7.4.1
        return np.where(self._frac_organic < self.PERCOLATION_THRESHOLD, 0,
            np.power(1 - self.PERCOLATION_THRESHOLD, -0.139) *\
                np.power(self._frac_organic - self.PERCOLATION_THRESHOLD, 0.139) *\
                self._frac_organic)

    @cached_property
    def _ksat(self):
        'Bulk saturated hydraulic conductivity of the soil (mm s-1)'
        # Equation 7.91 in CLM 4.0 Tech Note, Section 7.4.1
        f_uncon = 1 - self._frac_percolating
        return (f_uncon * self._ksat_uncon) +\
            ((1 - f_uncon) * self.params.ksat_om)

    @cached_property
    def _ksat_min(self):
        'Saturated hydraulic conductivity for mineral soil (mm s-1)'
        # Equation 7.90 in CLM 4.0 Tech Note, Section 7.4.1; multiply
        #   frac_sand by 100 because percent units expected
        return 0.0070556 * np.power(10, -0.884 + 0.0153 * self._frac_sand * 100)

    @cached_property
    def _ksat_uncon(self):
        'Hydraulic conductivity of the saturated, unconnected fraction (mm s-1)'
        # Equation 7.89 in CLM 4.0 Tech Note, Section 7.4.1
        f_uncon = 1 - self._frac_percolating
        return f_uncon * np.divide(1,
            np.divide(1 - self._frac_organic, self._ksat_min) +\
            np.divide(
                self._frac_organic - self._frac_percolating,
                self.params.ksat_om))

    @cached_property
    def _psi_sat(self):
        'Saturated matric potential, in millimeters (mm)'
        # Equation 7.86 in CLM 4.0 Tech Note, Section 7.4.1; weighted sum of
        #   saturated matric potential in mineral, organic fraction
        return ((1 - self._frac_organic) * self._psi_sat_min) +\
            (self._frac_organic * self.MATRIC_POTENTIAL_ORGANIC)

    @cached_property
    def _psi_sat_min(self):
        'Saturated matric potential of mineral soil, in millimeters (mm)'
        # Equation 7.87 in CLM 4.0 Tech Note, Section 7.4.1; multiply
        #   frac_sand by 100 because percent units expected
        return -10 * np.power(10, 1.88 - 0.0131 * self._frac_sand * 100)

    @cached_property
    def _root_fraction(self):
        'Soil root fraction in each layer'
        # Equation 2.11.1 in CLM 5.0 Tech Note, with values for beta taken
        #   from Jackson et al. (1996, Oecolegia), Table 1
        # ENF is average of boreal forest and temperature coniferous
        beta = np.array([ # Root extinction coefficient
            np.nan, 0.959, 0.962, 0.976, 0.966, 0.943, 0.964, 0.961, 0.961
        ])[self._pft]
        depth_cm = -(self._depths_m * 100) # Convert to cm and make positive
        return np.power(beta, np.vstack((np.zeros((1,1)), depth_cm[:-1]))) -\
            np.power(beta, depth_cm)

    @cached_property
    def _theta_sat(self):
        'Saturation water content (or saturation porosity)'
        # NOTE: Deviating from the approach in CLM 4.0; as we already know
        #   the soil's total porosity
        return np.array([self._porosity] * self._depths_m.size)\
            .reshape((self._depths_m.shape))

    @suppress_warnings
    def _potential_to_vwc(self, psi):
        'Convert a matric potential to a corresponding VWC'
        # Solve Equation 2.7.53 (in CLM 5.0 Tech Note) for theta
        theta_crit = np.power(
            self._theta_sat * (psi / self._psi_sat), -(1/self._b))
        return np.where(
            theta_crit > 1, 1, np.where(theta_crit < 0, 0, theta_crit))

    @cached_property
    def field_capacity(self):
        '''
        Critical point (VWC) or field capacity of the soil, conventionally
        defined as -0.033 MPa. Returns the equivalent volumetric water content
        (m3 m-3).
        '''
        # After Verhoef & Egea (2014), calculate the field capacity,
        #   "generally" at -0.033 MPa; first, define the critical point in
        #   terms of mm of potential;
        psi_crit = -0.033e6 / 9.8 # 1 mm of hydraulic head == 9.8 Pa
        return self._potential_to_vwc(psi_crit)

    @cached_property
    def field_capacity_balland(self):
        r'''
        Field capacity of the soil, from Balland et al. (2008).

        $$
        \theta_{FC} = \phi \times
        \left(c + (d - c)\sqrt{f_{clay}}\right) \times
        \mathrm{exp}\left(-\frac{a\times f_{sand} - b\times f_{om}}{\phi}\right)
        $$
        Where `f_clay`, `f_sand`, and `f_om` are the clay, sand, and organic
        material fractions of the soil; `phi` is the saturation porosity.
        '''
        # Equation 19 in Balland et al. (2008) with coefficients from Table 7
        return self._theta_sat *\
            (0.565 + ((0.991 - 0.565) * np.sqrt(self._frac_clay))) *\
            np.exp(-np.divide(
                (0.103 * self._frac_sand) - (0.785 * self._frac_organic),
                self._theta_sat))

    @cached_property
    def wilting_point(self):
        '''
        Permanent wilting point of the soil, conventionally defined as
        -1.5 MPa (Tolk et al. 2003). Returns the equivalent volumetric water
        content (m3 m-3).
        '''
        psi_wilt = -1.5e6 / 9.8 # 1 mm of hydraulic head == 9.8 Pa
        return self._potential_to_vwc(psi_wilt)

    @cached_property
    def wilting_point_balland(self):
        r'''
        Permanent wilting point of the soil, from Balland et al. (2008).

        $$
        \theta_{WP} = \theta_{FC} \times
        \left(c + (d - c)\sqrt{f_{clay}}\right) \times
        \mathrm{exp}\left(
        -\frac{a\times f_{sand} - b\times f_{om}}{\theta_{FC}}\right)
        $$
        '''
        fc = self.field_capacity_balland
        # Equation 20 in Balland et al. (2008) with coefficients from Table 7
        return fc * (0.17 + ((0.832 - 0.17) * np.sqrt(self._frac_clay))) *\
            np.exp(-np.divide(1.4 * self._frac_organic, fc))

    def f_ice(self, vwc, temp_k, alpha = 2, beta = 4):
        r'''
        The ice fraction of the combined liquid and ice water volumes, after
        the empirical formulation by Decker and Zeng (2006, Geophys. Res.
        Lett.). Formally, this is:

        $$
        f_{ice} = \frac{\theta_{ice}}{\theta_{ice} + \theta_{liq}}
        $$

        NOTE: The product of `f_ice` and the soil VWC (`theta`) is the
        volumetric ice content:

        $$
        \theta \times f_{ice} = \theta \frac{\theta_{ice}}{\theta_{ice} +
            \theta_{liq}} = \theta_{ice}
        $$

        Parameters
        ----------
        vwc : numpy.ndarray
            (Z x 1) array of soil volumetric water content (VWC)
        temp_k : numpy.ndarray
            (Z x 1) array of soil temperatures in degrees K
        alpha : int
            Empirical scaling parameter
        beta : int
            Empirical exponent parameter

        Returns
        -------
        numpy.ndarray
            (Z x 1) array of the volumetric ice fraction (dimensionless)
        '''
        wetness = vwc / self._theta_sat
        # Equation 4 in Decker and Zeng (2006); obtain ice as a fraction of
        #   the combined ice and liquid volumes (basically, the formula
        #   estimates how much of the combined water volume can be ice)
        f_ice = np.divide(
            1 - np.exp(alpha * np.power(wetness, beta) *\
                (temp_k - self.SOIL_FREEZING)),
            np.exp(1 - wetness))
        # Only valid for freezing temperatures; or, disregard values < 0,
        #   which occur above freezing
        f_ice = np.where(f_ice < 0, 0, f_ice)
        return np.where(f_ice > 1, 1, f_ice)

    def f_ice2(self, vwc, temp_k, field_capacity = None):
        r'''
        The ice fraction of the combined liquid and ice water volumes, after
        the empirical formulation by from the European Centre for Medium Range
        Weather Forecasting (ECMWF), as described by Decker and Zeng (2006,
        Geophys. Res. Lett.). This is a simplification, because it does not
        account for liquid water content.

        $$
        \theta_{ice} = \frac{\theta_{FC}}{2} \left[
        1 - \mathrm{sin}\left(\frac{\pi (T_K - 271.15)}{4}\right)
        \right]
        $$

        Where `T_K` is the temperature in degrees K, `theta_FC` is the field
        capacity. **The return value is the ice fraction, i.e.:**

        $$
        f_{ice} = \frac{\theta_{ice}}{\theta_{ice} + \theta_{liq}}
        $$

        Parameters
        ----------
        vwc : numpy.ndarray
            (Z x 1) array of soil volumetric water content (VWC)
        temp_k : numpy.ndarray
            (Z x 1) array of soil temperatures in degrees K
        field_capacity : float or None
            The field capacity (m3 m-3); if None, defaults to the definition
            from Balland et al. (2008)

        Returns
        -------
        numpy.ndarray
            (Z x 1) array of the volumetric ice fraction (dimensionless)
        '''
        if field_capacity is None:
            field_capacity = self.field_capacity_balland
        # Equation 2 in Decker and Zeng (2006)
        vwc_ice = (field_capacity / 2) * (1 -\
            np.abs(np.sin(np.divide(np.pi * (temp_k - self.SOIL_FREEZING - 2), 4))))
        vwc_ice = np.where(vwc_ice < 0, 0, vwc_ice)
        vwc_ice = np.where(
                temp_k > self.SOIL_FREEZING + 1, 0,
            np.where(
                temp_k < self.SOIL_FREEZING - 3, field_capacity, vwc_ice))
        return (vwc_ice / vwc)

    def f_impedance(self, vwc, f_ice):
        r'''
        Ice impedance of the soil layers.

        $$
        \Theta_{ice} = 10^{-\Omega F_{ice}}
        \quad\mbox{where}\quad F_{ice} = \theta\frac{f_{ice}}{\theta_{sat}}
            = \frac{\theta_{ice}}{\theta_{sat}};\,
        \Omega = 6
        $$

        Parameters
        ----------
        vwc : numpy.ndarray
            (Z x 1) array of soil volumetric water content (VWC)
        f_ice : numpy.ndarray
            (Z x 1) array of the ice fraction

        Returns
        -------
        numpy.ndarray
            (Z x 1) array of ice impedance
        '''
        # Equation 2.7.48 in CLM 5.0 Tech Note
        return np.power(10, -6 * ((vwc * f_ice) / self._theta_sat))

    def h2o_conductivity(self, vwc, f_ice):
        r'''
        Hydraulic conductivity (mm s-1) of each soil layer, as a function of
        the soil water and ice volumes.

        $$
        k_i = \left\{\begin{array}{lr}
        \Theta_{ice}\, k_{sat} \left(
            \frac{0.5(\theta_i + \theta_{i+1})}{0.5(\theta_{sat,i} +
                \theta_{sat,(i+1)})}\right)^{2B_i + 3}
        & 1\le i\le N -1\\\\
        \Theta_{ice}\, k_{sat}
        \left(\frac{\theta_i}{\theta_{sat,i}}\right)^{2B_i + 3}
        & i = N
        \end{array}\right\}
        $$

        Where `k_sat` is the saturated hydraulic conductivity, `B` is the
        Clapp & Hornberger exponent:

        $$
        B_i = B_{min,i}(1 - f_{om,i}) + 2.7(f_{om,i})
        \quad\mbox{where}\quad B_{min,i} = 2.91 + 0.159\times [\mathrm{Clay\%}]_i
        $$

        Parameters
        ----------
        vwc_liq : numpy.ndarray
            (Z x 1) array of soil volumetric water content (VWC)
        f_ice : numpy.ndarray
            (Z x 1) array of the ice fraction

        Returns
        -------
        numpy.ndarray
            (Z x 1) of hydraulic conductivity in each layer (mm s-1)
        '''
        # Calculate the liquid volumetric soil moisture
        vwc_liq = vwc * (1 - f_ice)
        impedance_i = self.f_impedance(vwc, f_ice)
        impedance_n = impedance_i[-1]
        b_exp = 2 * self._b + 3
        # NOTE: Operations on arrays [:-1] and [1:] operate on current layer
        #   and layer below, respectively, as bottom-most layer is excluded;
        #   shape (Z,N)
        # NOTE: Because saturation porosity is same for all layers,
        #   denominator of VWC contrast is merely self._theta_sat[1:], rather
        #   than an average of the porosity of this layer and the next
        k = impedance_i[:-1] * self._ksat[:-1] * np.power(
                np.divide(
                    0.5 * (vwc_liq[:-1] + vwc_liq[1:]), self._theta_sat[1:]
            ), b_exp[:-1])
        # Hydraulic conductivity of the bottom-most layer; shape (N,)
        kn = impedance_n * self._ksat[-1] *\
            np.power(np.divide(vwc_liq[-1], self._theta_sat[-1]), b_exp[-1])
        return np.vstack((k, kn[np.newaxis,...]))

    def matric_potential(self, vwc, f_ice):
        r'''
        The soil matric potential (mm), defined at the "node depth," or at the
        midpoint of the soil layer.

        $$
        \psi_i = \psi_{sat,i}\left(
          \frac{\theta_i}{\theta_{sat,i}}
        \right)^{-B_i}
        \quad\mbox{where}\quad \psi_i \ge -1\times 10^8;\quad
        0.01 \le \frac{\theta_i}{\theta_{sat,i}} \le 1
        $$

        Where `psi_sat` is the saturated soil matric potential, `B` is the
        Clapp & Hornberger exponent; see `SoilProfile.h2o_conductivity()`.

        Parameters
        ----------
        vwc_liq : numpy.ndarray
            (Z x 1) array of soil volumetric water content (VWC)
        f_ice : numpy.ndarray
            (Z x 1) array of the ice fraction

        Returns
        -------
        numpy.ndarray
            (Z x 1) array of soil matric potential
        '''
        # Calculate the liquid volumetric soil moisture
        vwc_liq = vwc * (1 - f_ice)
        # Equation 2.7.53 in CLM 5.0 Tech Note
        quo = np.divide(vwc_liq, self._theta_sat)
        quo = np.where(quo < 0.01, 0.01, np.where(quo > 1, 1, quo))
        psi0 = self._psi_sat * np.power(quo, -self._b)
        return np.where(psi0 < -1e8, -1e8, psi0)

    def max_infiltration(self, vwc, temp_k, f_saturated, f_ice = None):
        r'''
        The maximum infiltration capacity of the (surface) soil layer.

        $$
        q_{max} = (1 - f_{sat}) \Theta_{ice} k_{sat}
        $$

        Where `f_sat` is the fraction of the land surface that is saturated,
        `Theta_ice` is the impedance due to ice, and `k_sat` is the saturated
        hydraulic conductivity.

        Parameters
        ----------
        vwc : numpy.ndarray
            (Z x 1) array of soil volumetric water content (VWC)
        temp_k : numpy.ndarray
            (Z x 1) array of soil temperatures in degrees K
        f_saturated : numpy.ndarray
            (Z x 1) array of the fraction of the land surface that is
            saturated
        f_ice : numpy.ndarray or None
            (Optional) (Z x 1) array of the ice fraction; will be calculated
            based on VWC and temperature if None

        Returns
        -------
        numpy.ndarray
            The maximum infiltration capacity (kg m-2 s-1); array of shape (N,)
        '''
        if f_ice is None:
            f_ice = self.f_ice(vwc, temp_k)
        impedance_i = self.f_impedance(vwc, f_ice)
        # Equation 2.7.34 in CLM 5.0 Tech Note
        return (1 - f_saturated) * impedance_i[0] * self._ksat[0]

    def solve_sink(
            self, vwc, transpiration, q = 1, use_balland = True,
            clip_to_saturation = True):
        '''
        Calculates hydraulic sink term for each soil layer. Currently, this
        is limited to transpiration from each soil layer. In the future, this
        may possibly include evaporation from the soil surface. The soil water
        soil water stress constraint is estimated as described in:

        Verhoef, A., & Egea, G. (2014). Modeling plant transpiration under
          limited soil water: Comparison of different plant and soil
          hydraulic parameterizations and preliminary implications for
          their use in land surface models. *Agricultural and Forest
          Meteorology*, 191, 22–32.

        NOTE: This function does not distinguish between liquid VWC and the
        overall (ice and water) VWC. This is for simplicity, and because it is
        assumed that potential transpiration is already limited by
        temperature. Increasing transpiration (e.g., by setting q to a value
        less than 1) will increase the magnitude of dry-down but not
        necessarily decrease peak soil moisture in near-surface layers during
        infiltration events.

        Parameters
        ----------
        vwc : numpy.ndarray
            Current soil moisture (VWC) in each soil layer
        transpiration : float
            Total potential (unconstrained) transpiration rate (kg m-2 s-1),
            a daily scalar value
        q : float
            Curvature exponent for the soil water stress factor (Default: 1)
        use_balland : bool
            True to use the self-consistent formulae for field capacity and
            wilting point from Balland et al. (2008); False to define those
            based on soil matric potentials of -0.033 MPa and -1.5 MPa,
            respectively (Default: True)
        clip_to_saturation : bool
            True to force field capacity to be no larger than the saturation
            porosity (takes the minimum of field capacity and saturation
            porosity) (Default: True)

        Returns
        -------
        tuple
            A 3-tuple of (transpiration, soil_evaporation,
            canopy_evaporation) arrays; transpiration is a (Z x 1) array and
            the other two elements are currently `None`.
        '''
        # Leaf conductance to sensible heat; there are just two unique values
        #   for g_h in MODIS MOD16 Collection 6.1 (User's Guide, Table 3.2)
        # g_h = 0.01 if self._pft <= 4 else 0.02
        # And the conductance to evaporated water vapor per unit LAI just
        #   happens to be the same
        # g_e = g_h
        # TODO Wet canopy evaporation (kg m-2 s-1)
        # canopy_evap = canopy_evaporation(
        #     pressure, air_temp_k, rhumidity, vpd, lai, fpar, rad_canopy,
        #     g_h = g_h, g_e = g_e)
        # Soil moisture stress (Verhoef & Egea, 2014)
        fc = self.field_capacity_balland
        wp = self.wilting_point_balland
        if not use_balland:
            fc = self.field_capacity
            wp = self.wilting_point
        if clip_to_saturation:
            fc = np.where(fc > self._theta_sat, self._theta_sat, fc)
        stress = np.divide(vwc - wp, fc - wp)
        stress = np.where(stress > 1, 1, np.where(stress < 0, 0, stress))
        # Product of root fraction in each layer, plant water stress, and the
        #   total potential transpiration; divide by latent heat of
        #   vaporization to convert to a mass flux
        trans_i = np.power(stress, q) * self._root_fraction * transpiration
        # TODO Soil evaporation is the residual of ET minus transpiration and
        #   wet canopy evaporation
        # soil_evap = et - np.sum(trans_i, axis = 0) - canopy_evap
        # return (trans_i, soil_evap, canopy_evap)
        return (trans_i, None, None)

    def solve_vwc(
            self, influx, vwc, temp_k, dt, transpiration = None,
            saturated_below = False):
        r'''
        Solves for volumetric water content (VWC) at a single time step for
        each depth using a tridiagonal system of equations for the water
        balance. The free drainage ("flux") bottom boundary condition is
        always enforced because, otherwise, soil layers will saturate quickly;
        the aquifer is uncoupled and assumed to lie below the soil layer.
        Other considerations:

        1. Below the soil profile there is no ice (ice-filled fraction is
            zero); this is out of necessity in calculating derivatives
            but is also consistent with a geothermal heat flux that
            maintains above-freezing conditions below the profile.
        2. Sub-surface runoff is computed separately; it is one of the values
            returned by this function and should be subtracted from the soil
            VWC profile after updating with the change in VWC estimated by
            this function.
        3. Similarly, if there is a perched, saturated layer above a frozen
            layer, the returned value of "runoff" includes lateral drainage
            from the perched layer(s), which should also be subtracted from
            the soil VWC profile.

        At a minimum, runoff includes drainage according to the free-drainage
        or "flux" bottom boundary condition of CLM 5.0:

        $$
        q_{drain} = k_i + \left[
        \frac{\partial\, k}{\partial\, \theta_{liq}} \times \Delta \theta_{liq}
        \right]_i
        $$

        Where `k` is the hydraulic conductivity. If saturated conditions exist
        within the soil column (saturated from the bottom-up), then additional,
        lateral sub-surface runoff is calculated as described in CLM 4.0
        Technical Note, Section 7.5:

        $$
        q_{drain} = \Theta_{ice}\, 10\,\mathrm{sin}(\beta )\,
            \mathrm{exp}(-f_{drain} z_{\nabla})
            \quad\mbox{where}\quad f_{drain} = 2.5\,\mathrm{m}^{-1}
        $$

        Where `beta` is the slope, `z_nabla` is the depth to the top of the
        saturated zone, and `Theta_ice` is the impedance due to ice. The
        specific yield is calculated:

        $$
        S_y = \theta_{sat}\left(1 - \left(
        1 + \frac{z_{\nabla}}{\Psi_{sat}}
        \right)^{-1/B}\right)
        $$

        Parameters
        ----------
        influx: float or numpy.ndarray
            Scalar or N-dimensional array of water influx at the surface
            layer, in units of (kg m-2 s-1) or (mm s-1), as 1 mm of water
            over an area of 1 m-2 weighs 1 kg.
        vwc : numpy.ndarray
            (Z x 1) array of total soil volumetric water content (VWC) for the
            current time step, including both liquid and ice water content
        temp_k : numpy.ndarray
            (Z x 1) array of soil temperatures in degrees K for the current
            time step
        dt : int
            Size of time step (secs)
        transpiration : numpy.ndarray
            (Optional) Transpiration in each soil layer (kg m-2 s-1), a
            (Z x 1) array
        saturated_below : bool
            True to invoke a virtual soil layer below the soil profile that
            is fully saturated (Default: False)

        Returns
        -------
        tuple
            Returns a 3-tuple of (`solution`, `flows`, `runoff`) where
            `solution` is the change in VWC in each layer; flows is a tuple of
            (`q_in`, `q_out`) where `q_in` is the flow into each layer from
            the layer above, and `q_out` is the flow out of each layer (all
            flows in units of mm s-1); `runoff` is the change in VWC due to
            lateral sub-surface runoff.
        '''
        @suppress_warnings
        def _dk_dliq(vwc_liq, temp_k, mean_impedance, bottom_vwc_liq):
            # Equation 2.7.88 in CLM 5.0 Tech Note: d(k_i)/d(VWC_i),
            #   valid for all layers (with assumptions); same equation for:
            #       d(k_i)/d(VWC_i)
            #       d(k_i)/d(VWC_j), j = i + 1
            # Average of this layer's VWC and the one below (Z layers)
            mean_vwc_liq = np.vstack([
                0.5 * (vwc_liq[:-1] + vwc_liq[1:]),
                0.5 * (vwc_liq[-1] + bottom_vwc_liq)
            ])
            # And we don't average the saturation porosity (theta_sat) because
            #   it is the same for all layers
            result = (2 * self._b + 3) * mean_impedance *\
                self._ksat * np.power(
                    mean_vwc_liq / self._theta_sat, 2 * self._b + 2) *\
                (0.5 / self._theta_sat)
            return result

        def _drain_runoff(vwc, solution, mean_impedance, dk_dliq):
            'Compute specific yield, drainage from lateral sub-surface runoff'
            drain_runoff = self._zeros.copy()
            sp_yield = np.inf
            # If bedrock is within the soil column, there should be no free
            #   drainage; otherwise...
            if self._z_bedrock < self._z[-1]:
                # Drainage from the bottom layer according to the "flux"
                #   bottom boundary condition of CLM 5.0 Fortran code, ca.
                #   Line 1383 of SoilWaterMovementMod.F90
                drain_runoff[-1] = k[-1] + (dk_dliq[-1] * solution[-1])
            # In addition, there may be lateral drainage from the bottom layer
            #   due to saturation within the soil column
            sat_mask = np.full(vwc.shape, False)
            # Layers are "saturated" if >/= 90% of saturation porosity
            for i in range(0, vwc.size):
                j = vwc.size - i - 1
                if vwc[j] >= 0.9 * self._theta_sat[j]:
                    sat_mask[j] = True
                else:
                    break # Stop at the first layer not saturated
            if sat_mask.any():
                # Water table is at the top of the saturated soil layers; units
                #   are in meters because DRAINAGE_DECAY_FACTOR is in m-1
                table_depth = self._depths_m[:-1][sat_mask[1:]]
                # Unless the entire column is saturated...
                if table_depth.size > 0:
                    # Equation 7.170 in CLM 4.5 Tech Note;
                    #   NOTE: table_depth.max() here means shallowest layer
                    #   that is completely saturated
                    drain_runoff[-1] = mean_impedance[sat_mask].mean() * 10 *\
                        np.sin(self._slope) *\
                        np.exp(-self.DRAINAGE_DECAY_FACTOR * table_depth.max())
                    # Equation 7.174 in CLM 4.5 Tech Note or
                    #   2.7.110 in CLM 5.0 Tech Note;
                    sp_yield = self._theta_sat[sat_mask].mean() * (1 - np.power(
                        1 + (table_depth.max() / self._psi_sat[sat_mask].mean()),
                        -1 / self._b[sat_mask].mean()))
            return (drain_runoff, sp_yield)

        def _drain_perched(vwc, temp_k, f_ice):
            '''
            Compute drainage from a perched saturated zone (if any); the
            perched zone must be >/= 90% saturated and lie above a frozen
            layer.
            '''
            drain_perched = self._zeros.copy()
            sp_yield = np.inf
            perched = np.logical_and(
                vwc[:-1] >= 0.9 * self._theta_sat[:-1], f_ice[1:] > 0)
            if perched.any():
                # Identify shallowest frozen layer with unfrozen layer above,
                #   which may NOT include the surface layer
                i_perch = np.argwhere(perched[:,0]).min()
                i_frost = np.argwhere(perched[:,0]).max() + 1
                impedance = self.f_impedance(vwc, f_ice)
                # Equation 2.7.107 in CLM 5.0 Tech Note
                ksat_perch = 10e-5 * np.sin(self._slope) * np.divide(
                    (impedance * self._ksat * self._thickness_mm)[
                        i_perch:(i_frost + 1)
                    ].sum(),
                    self._thickness_mm[i_perch:(i_frost + 1)].sum())
                drain_perched[i_frost - 1] = ksat_perch * (
                    self._depths_m[i_frost - 1] - self._depths_m[i_perch - 1]
                ) * 1e3 # Convert from m to mm
                # Equation 7.174 in CLM 4.5 Tech Note or
                #   2.7.110 in CLM 5.0 Tech Note;
                perch_mask = np.vstack((perched, False))
                sp_yield = self._theta_sat[perch_mask].mean() * (1 - np.power(
                    1 + ((self._depths_m[i_frost - 1] * 1e3)\
                        / self._psi_sat[perch_mask].mean()),
                    -1 / self._b[perch_mask].mean()))
            return (drain_perched, sp_yield)

        # To speed things up, if temps. well above freezing, f_ice = 0
        if (temp_k > 276).all():
            f_ice = np.zeros((vwc.shape))
        else:
            f_ice = self.f_ice(vwc, temp_k)
        # Calculate the liquid volumetric soil moisture and the volumetric
        #   ice content
        vwc_liq = vwc * (1 - f_ice)
        vwc_ice = vwc * f_ice
        # Average of this layer's ice-filled fraction and the one below
        mean_f_ice = 0.5 * (
            np.vstack((vwc_ice[1:], np.zeros(vwc_ice.shape)[0:1])) + vwc_ice)
        mean_impedance = self.f_impedance(vwc, mean_f_ice)
        psi = self.matric_potential(vwc, f_ice)
        k = self.h2o_conductivity(vwc, f_ice)
        nans = np.ones(k[0].shape) * np.nan # Single layer of NaNs

        # Compute derivatives
        if saturated_below:
            # Assume that a virtual soil layer below the soil profile is fully
            #   saturated
            dk_dliq = _dk_dliq(
                vwc_liq, temp_k, mean_impedance,
                bottom_vwc_liq = np.mean((vwc_liq[-1], self._theta_sat[-1])))
        else:
            # Otherwise, allow mean VWC is that of the bottom layer
            dk_dliq = _dk_dliq(
                vwc_liq, temp_k, mean_impedance, bottom_vwc_liq = vwc_liq[-1])
        dk0_dliq = np.vstack((np.nan, dk_dliq[:-1]))

        # Equation 2.7.84-2.7.86 in CLM 5.0 Tech Note: d(psi_j)/d(VWC_j),
        #   for all j in (i-1, i, i+1) where i = z in N; e.g., for j = i-1,
        #   index at (i-1)
        dpsi_dliq = np.where(vwc_liq < 0.01 * self._theta_sat, 0.01 * self._theta_sat,
            np.where(vwc_liq > self._theta_sat, self._theta_sat,
            -self._b * (psi / vwc_liq)))

        # Equation 2.7.80 in CLM 5.0 Tech Note: d(q_j)/d(VWC_j),
        #   only valid for layers 0 < i < N; j = i - 1
        n_diff = self._z_node - np.vstack((np.nan, self._z_node[:-1]))
        dqout0_dliq0 = -((np.vstack((nans, k[:-1])) / n_diff) *\
            np.vstack((nans, dpsi_dliq[:-1]))) -\
            dk0_dliq *\
            (((np.vstack((nans, psi[:-1])) - psi) + n_diff) / n_diff)

        # Equation 2.7.81 in CLM 5.0 Tech Note: d(q_j)/d(VWC_i),
        #   only valid for layers 0 < i; j = i - 1
        dqout0_dliq = ((np.vstack((nans, k[:-1])) / n_diff) * dpsi_dliq) -\
            dk0_dliq *\
            (((np.vstack((nans, psi[:-1])) - psi) + n_diff) / n_diff)
        # Taken from the CLM 5.0 Fortran code, SoilWaterMovementMod.F90,
        #   ca. Line 1663 (boundary condition for surface layer); BUT it is
        #   not necessary to set here, as term is not used for surface layer
        # dqout0_dliq[0] = 0

        # Equation 2.7.82 in CLM 5.0 Tech Note: d(q_i)/d(VWC_i),
        #   only valid for layers i < N
        dqout_dliq = np.vstack((dqout0_dliq0[1:], np.nan,))
        # Taken from the CLM 5.0 Fortran code, SoilWaterMovementMod.F90,
        #   ca. Line 1831
        dqout_dliq[-1] = dk_dliq[-1] / self._theta_sat[-1]

        # Equation 2.7.83 in CLM 5.0 Tech Note: d(q_i)/d(VWC_j),
        #   only valid for layers i < N; j = i + 1
        dqout_dliq1 = np.vstack((dqout0_dliq[1:], np.nan))

        # Calculate flows from this layer (q_out) and one above (q_in); can
        #   calculate a single contrast term because difference in psi is
        #   always top-to-bottom; diff. in node depth always bottom-to-top
        contrast = np.divide(
            (psi[:-1] - psi[1:]) + (self._z_node[1:] - self._z_node[:-1]),
            (self._z_node[1:] - self._z_node[:-1]))
        # Equation 2.7.79 in CLM 5.0 Tech Note
        q_out = np.vstack((-k[:-1] * contrast, -k[-1]))
        # Equation 2.7.78 in CLM 5.0 Tech Note;
        #   NOTE: This could be -k[1:] * contrast, instead, and would still
        #   work at finer time steps, without transpiration loss
        q_in = np.vstack((influx, -k[:-1] * contrast))

        # Create a (Z x 3) system of equations: a*X1 + b*X2 + c*X3 = r
        nn = self._depths_m.size
        lhs = np.zeros((nn, nn))
        rhs = np.ones((nn,)) * np.nan
        n_layers = len(self._depths_m)
        for z, depth in enumerate(self._depths_m):
            # Calculate LHS coefficients of the tridiagonal equation
            if z == 0:
                # Soil layer i = 0, Equations 2.7.90 - 2.7.93 (CLM 5.0);
                #   a = 0
                b = dqout_dliq[z] - (self._thickness_mm / dt)[z]
                c = dqout_dliq1[z]
                lhs[z,0:2] = np.array((b, c)).ravel()
            elif z < (n_layers - 1):
                # Soil layers 0 < i < N, Equations 2.7.94 - 2.7.97 (CLM 5.0)
                a = -dqout0_dliq0[z]
                b = dqout_dliq[z] - dqout0_dliq[z] - (self._thickness_mm / dt)[z]
                c = dqout_dliq1[z]
                lhs[z,(z-1):(z+2)] = np.array((a, b, c)).ravel()
            else:
                # Soil layer i = N, Equations 2.7.98 - 2.7.101 (CLM 5.0);
                #   c = 0 (q_out in this layer is zero)
                a = -dqout0_dliq0[z]
                b = dqout_dliq[z] - dqout0_dliq[z] - (self._thickness_mm / dt)[z]
                lhs[z,-2:] = np.array((a, b)).ravel()
            # Calculate RHS of the tridiagonal equation
            if transpiration is None:
                r = q_in[z] - q_out[z]
            else:
                # Despite Equation 2.7.77, the hydraulic sink term is indeed
                #   subtracted in this equation in the CLM 5.0 Fortran code
                r = q_in[z] - q_out[z] - transpiration[z]
            rhs[z] = r.ravel()
        # Solve tridiagonal system
        banded = np.vstack(( # Creating banded matrix faster this way
            np.hstack((np.diag(lhs[1:,]), 0)),
            np.diag(lhs),
            np.hstack((0, np.diag(lhs[:,1:])))))
        solution = tridiag_solver(lhs, rhs, banded = banded)[:,np.newaxis]
        # Compute lateral sub-surface drainage from saturated zone; convert
        #   to (change in) VWC and compare to specific yield
        q_runoff, sp_yield = _drain_runoff(
            vwc, solution, mean_impedance, dk_dliq)
        runoff = np.abs(q_runoff) * (dt / self._thickness_mm)
        if np.isfinite(sp_yield):
            runoff = np.where(np.abs(runoff) > sp_yield, sp_yield, runoff)
        # Compute lateral sub-surface drainage from perched zone; convert
        #   to (change in) VWC and compare to specific yield
        q_perched, sp_yield = _drain_perched(vwc, temp_k, f_ice)
        drainage = np.abs(q_perched) * (dt / self._thickness_mm)
        if np.isfinite(sp_yield):
            drainage = np.where(np.abs(drainage) > sp_yield, sp_yield, drainage)
        return (solution, (q_in, q_out), runoff + drainage)
