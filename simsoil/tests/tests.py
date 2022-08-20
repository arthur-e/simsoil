'''
Test cases for the `simsoil` library.
'''

import os
import unittest
import numpy as np
from simsoil.core import DEPTHS, SOC_RATIOS, InfiltrationModel, SoilProfile

np.seterr(all = 'ignore')

class SoilWaterInfiltrationTestSuite(unittest.TestCase):
    '''
    Integration tests for the InfiltrationModel and SoilProfile classes.
    '''
    defaults = {
        'soc': SOC_RATIOS,
        'depths_m': DEPTHS,
        'bedrock': -3,
    }
    soils = [
        SoilProfile( # Sandy soils, low porosity, low slope
            1, sand = 0.6, clay = 0.1, porosity = 0.3, slope = 0.01,
            **defaults),
        SoilProfile( # Sandy soils, high porosity, low slope
            1, sand = 0.6, clay = 0.1, porosity = 0.5, slope = 0.01,
            **defaults),
        SoilProfile( # Clay soils, low porosity, low slope
            1, sand = 0.1, clay = 0.6, porosity = 0.3, slope = 0.01,
            **defaults),
        SoilProfile( # Clay soils, high porosity, low slope
            1, sand = 0.1, clay = 0.6, porosity = 0.5, slope = 0.01,
            **defaults),
    ]
    models = [
        InfiltrationModel(s)
        for s in soils
    ]
    models_drained = [ # With "runoff" bottom boundary condition
        InfiltrationModel(
            soil_model = SoilProfile(
                1, sand = 0.1, clay = 0.6, porosity = 0.3, slope = 2,
                **defaults)),
        InfiltrationModel(
            soil_model = SoilProfile(
                1, sand = 0.1, clay = 0.6, porosity = 0.3, slope = 2,
                **defaults)),
    ]
    models_transpiration = [
        InfiltrationModel(
            soil_model = SoilProfile(
                4, sand = 0.6, clay = 0.1, porosity = 0.35, slope = 1,
                **defaults)),
        InfiltrationModel(
            soil_model = SoilProfile(
                5, sand = 0.6, clay = 0.1, porosity = 0.35, slope = 1,
                **defaults)),
    ]

    @classmethod
    def setUpClass(cls):
        x = np.arange(0, 365)
        np.random.seed(9)
        cls.influx = 1e-5 + (np.random.sample(x.size) * 5e-6) +\
            1e-5 * np.sin(x * (np.pi / 270))
        cls.influx_low = 2e-6 + (np.random.sample(x.size) * 5e-7) +\
            1e-6 * np.sin(x * (np.pi / 270))
        cold = 270 + (np.random.sample(x.size) * 3) + 5 * np.sin(x * (np.pi / x.size))
        cls.temp_profile_cold = np.vstack([
            (-1, 0, 1, 1, 2, 2)[i] + cold for i in range(0, DEPTHS.size)
        ])
        warm = 290 + np.random.sample(x.size) + 5 * np.sin(x * (np.pi / x.size))
        cls.temp_profile_warm = np.vstack([
            (-1, 0, 1, 1, 2, 2)[i] + warm for i in range(0, DEPTHS.size)
        ])
        cls.potential_transpiration = 1e-4 + (4e-5 * np.random.sample(365)) +\
            1e-4 * np.sin(x * (np.pi / 290))

    def test_soils_with_warm_temps(self):
        'Tests model with "flux" boundary condition, warm temperatures'
        init_vwc = np.ones(DEPTHS.shape) * 0.15
        f_wet = np.ones(self.__class__.influx.size) * 0.2
        for i, model in enumerate(self.__class__.models):
            results, _, _ = model.run(
                init_vwc, self.__class__.temp_profile_warm,
                transpiration = None, influx = self.__class__.influx,
                dt = 7200, f_saturated = f_wet, adaptive = False)
            self.assertEqual(results.size, DEPTHS.size * f_wet.size)
            # Texture:  [Sandy, Sandy, Clay, Clay]
            # Porosity: [Low,    High,  Low, High]
            self.assertEqual(results.mean().round(3), (0.222, 0.315, 0.256, 0.323)[i])
            self.assertEqual( results.min().round(3), (0.150, 0.150, 0.150, 0.150)[i])
            self.assertEqual( results.max().round(3), (0.300, 0.500, 0.300, 0.422)[i])
            self.assertEqual( results.var().round(4), (0.0023, 0.0115, 0.0019, 0.0109)[i])

    def test_soils_with_cold_temps(self):
        'Tests model with "flux" boundary condition, cold temperatures'
        init_vwc = np.ones(DEPTHS.shape) * 0.15
        f_wet = np.ones(self.__class__.influx.size) * 0.2
        for i, model in enumerate(self.__class__.models):
            if i > 0:
                break # Skip clay and high-porosity models to save time
            results, _, _ = model.run(
                init_vwc, self.__class__.temp_profile_cold,
                transpiration = None, influx = self.__class__.influx,
                dt = 1800, f_saturated = f_wet, adaptive = True, ltol = 1e-3)
            self.assertEqual(results.size, DEPTHS.size * f_wet.size)
            # Texture:  [Sandy, None, None, None]
            # Porosity: [Low,   None, None, None]
            self.assertEqual(np.nanmean(results).round(3), 0.217)
            self.assertEqual( np.nanmin(results).round(3), 0.148)
            self.assertEqual( np.nanmax(results).round(3), 0.336)
            self.assertEqual( np.nanvar(results).round(4), 0.0024)

    def test_soils_with_transpiration_with_warm_temps(self):
        'Tests model with transpiration, warm temperatures'
        init_vwc = np.ones(DEPTHS.shape) * 0.15
        f_wet = np.ones(self.__class__.influx.size) * 0.1
        for i, model in enumerate(self.__class__.models_transpiration):
            results, _, _ = model.run(
                init_vwc, self.__class__.temp_profile_warm,
                transpiration = self.__class__.potential_transpiration,
                influx = self.__class__.influx, dt = 7200, f_saturated = f_wet,
                adaptive = False)
            self.assertEqual(results.size, DEPTHS.size * f_wet.size)
            self.assertEqual(results.mean().round(3), (0.104, 0.107)[i])
            self.assertEqual( results.min().round(3), (0.078, 0.078)[i])
            self.assertEqual( results.max().round(3), (0.150, 0.150)[i])
            self.assertEqual( results.var().round(5), (0.00065, 0.0007)[i])
            # i == 0 --> Deciduous Broadleaf
            # i == 1 --> Grassland
            if i == 0:
                self.assertTrue(np.equal(
                    results[:,-1].round(3), [0.119, 0.097, 0.079, 0.078, 0.086, 0.133]).all())
            if i == 1:
                self.assertTrue(np.equal(
                    results[:,-1].round(3), [0.112, 0.087, 0.078, 0.081, 0.118, 0.143]).all())


class InfiltrationModelTestSuite(unittest.TestCase):
    '''
    Unit tests for InfiltrationModel.
    '''
    _depths = -np.array((0.05, 0.1, 0.2, 0.4, 0.75, 1.5)).reshape((6,1))
    _soc = np.array((1, 1.77, 2.86, 4.63, 6.46, 9.56)).reshape((6,1)) * np.array(3138)
    _sm_profile = np.array([0.1374, 0.1368, 0.1357, 0.1336, 0.1297, 0.1270])\
        .reshape((6,1))
    _temp_profile = np.array([276, 277, 280, 284, 287, 290]).reshape((6,1))

    def test_infiltration_model_step_daily(self):
        'InfiltrationModel should calculate daily steps correctly'
        model = InfiltrationModel(
            soil_model = SoilProfile(
                1, self._soc, sand = 0.6, clay = 0.1, porosity = 0.4,
                bedrock = -1.8, slope = 0, depths_m = self._depths))
        # Hourly time steps
        final_vwc, de = model.step_daily(
            self._sm_profile, self._temp_profile, 0, 1e-6, 0.1, dt = 3600)
        self.assertEqual(final_vwc.sum().round(4), 0.7989)
        self.assertEqual(final_vwc[0].round(4), 0.1346)
        self.assertEqual(len(de), 23) # Initital error can't be calculated
        self.assertEqual(np.stack(de).max().round(5), 0.00838)
        self.assertEqual(np.stack(de).mean().round(5), -0.00061)
        # Smaller time steps
        final_vwc, de = model.step_daily(
            self._sm_profile, self._temp_profile, 0, 1e-6, 0.1, dt = 450)
        self.assertEqual(final_vwc.sum().round(4), 0.7989)
        self.assertEqual(final_vwc[0].round(4), 0.1346)
        self.assertEqual(np.stack(de).max().round(5), 0.00108)
        self.assertEqual(np.stack(de).mean().round(5), -8e-5)


class SoilProfileTestSuite(unittest.TestCase):
    '''
    Unit tests for SoilProfile.
    '''
    _bedrock = -1.8 # FIXME
    _depths = -np.array((0.05, 0.1, 0.2, 0.4, 0.75, 1.5)).reshape((6,1))
    _soc = np.array((1, 1.77, 2.86, 4.63, 6.46, 9.56)).reshape((6,1)) * np.array(3138)
    _sm_profile = np.array([0.1374, 0.1368, 0.1357, 0.1336, 0.1297, 0.1270])\
        .reshape((6,1))
    _temp_profile = np.array([276, 277, 280, 284, 287, 290]).reshape((6,1))

    def test_constants(self):
        'Should accurately calculate soil texture constants'
        model = SoilProfile(
            1, self._soc, sand = 0.6, clay = 0.1,
            porosity = 0.4, bedrock = -1.8, slope = 0, depths_m = self._depths)
        self.assertEqual(model._b.sum().round(3), 23.587)
        self.assertEqual(model._b.max().round(3), 4.223)
        self.assertEqual(model._frac_percolating.max(), 0)
        model = SoilProfile(
            1, SOC_RATIOS * np.array([5000]), sand = 0.6, clay = 0.1,
            porosity = 0.4, bedrock = -1.8, slope = 0, depths_m = self._depths)
        self.assertEqual(model._frac_percolating.max().round(3), 0.877)

    def test_hydraulic_constants(self):
        'Should accurately calculate saturated hydraulic conductivity'
        model = SoilProfile(
            1, self._soc, sand = 0.6, clay = 0.1,
            porosity = 0.4, bedrock = -1.8, slope = 0, depths_m = self._depths)
        self.assertEqual(model._ksat.sum().round(3), 0.066)
        self.assertEqual(model._ksat.max().round(3), 0.014)
        self.assertEqual(model._ksat_uncon.sum().round(3), 0.066)
        self.assertEqual(model._ksat_uncon.max().round(3), 0.014)
        self.assertEqual(model._ksat_min.max().round(3), 0.008)
        model = SoilProfile(
            1, SOC_RATIOS * np.array([5000]), sand = 0.6, clay = 0.1,
            porosity = 0.4, bedrock = -1.8, slope = 0, depths_m = self._depths)
        self.assertEqual(model._ksat.sum().round(3), 0.322)
        self.assertEqual(model._ksat.max().round(3), 0.089)
        self.assertEqual(model._ksat_uncon.sum().round(3), 0.063)
        self.assertEqual(model._ksat_uncon.max().round(3), 0.014)

    def test_matric_potential_constants(self):
        'Should accurately calculate saturated matric potential'
        model = SoilProfile(
            1, self._soc, sand = 0.6, clay = 0.1,
            porosity = 0.4, bedrock = -1.8, slope = 0, depths_m = self._depths)
        self.assertEqual(model._psi_sat.sum().round(1), -529.1)
        self.assertEqual(model._psi_sat.max().round(1), -69.2)
        self.assertEqual(model._psi_sat_min.max().round(1), -124.2)
        model = SoilProfile(
            1, self._soc, sand = 0.06, clay = 0.6,
            porosity = 0.4, bedrock = -1.8, slope = 0, depths_m = self._depths)
        self.assertEqual(model._psi_sat.sum().round(1), -2617.1)
        self.assertEqual(model._psi_sat.max().round(1), -332.4)
        self.assertEqual(model._psi_sat_min.max().round(1), -633.0)

    def test_f_ice(self):
        'Should accurately calculate ice fraction'
        model = SoilProfile(
            1, self._soc, sand = 0.6, clay = 0.1,
            porosity = 0.4, bedrock = -1.8, slope = 0, depths_m = self._depths)
        f_ice = model.f_ice(self._sm_profile, self._temp_profile)
        self.assertTrue(np.equal(f_ice, np.zeros((6,1))).all())
        f_ice = model.f_ice(self._sm_profile, self._temp_profile - 4)
        self.assertEqual(f_ice[0].round(3), 0.016)
        self.assertEqual(f_ice[1].round(3), 0.002)
        self.assertEqual(f_ice.sum().round(4), 0.0185)
        # Change up the soil texture
        model = SoilProfile(
            1, self._soc, sand = 0.1, clay = 0.5,
            porosity = 0.2, bedrock = -1.8, slope = 0, depths_m = self._depths)
        f_ice = model.f_ice(self._sm_profile, self._temp_profile - 4)
        self.assertEqual(f_ice.sum().round(4), 0.3395)

    def test_maximum_infiltration(self):
        'Should accurately estimate maximum water infiltration'
        model = SoilProfile(
            1, self._soc, sand = 0.6, clay = 0.1,
            porosity = 0.4, bedrock = -1.8, slope = 0, depths_m = self._depths)
        max_infil = model.max_infiltration(0.15, 280, 0.1)
        self.assertEqual(max_infil[0].round(4), 0.0124)
        max_infil = model.max_infiltration(0.4, 280, 0.1)
        self.assertEqual(max_infil[0].round(4), 0.0124)
        max_infil = model.max_infiltration(0.4, 273, 0.1)
        self.assertEqual(max_infil[0].round(4), 0.0003)
        max_infil = model.max_infiltration(0.15, 280, 0.4)
        self.assertEqual(max_infil[0].round(4), 0.0083)

    def test_soil_profile_flows(self):
        'Should accurately simulate soil water flows between layers'
        f_clay = 0.06667
        f_sand = 0.66667
        table_depth = -1.8
        porosity = np.array(0.4056)
        model = SoilProfile(
            1, self._soc, f_sand, f_clay, porosity, self._bedrock, 0,
            self._depths)
        dt = 500
        time_steps = np.arange(dt, 86400, dt)
        # With default influx
        all_q_in = []
        all_q_out = []
        for t, _ in enumerate(time_steps):
            if t == 0:
                vwc = self._sm_profile
            d_vwc, flows, _ = model.solve_vwc(1.4e-10, vwc, self._temp_profile, dt = dt)
            q_in, q_out = flows
            all_q_in.append(q_in)
            all_q_out.append(q_out)
            vwc = np.add(vwc, d_vwc)
        self.assertTrue(np.equal(
            np.stack(all_q_in).sum(axis = 0).round(5).ravel(), np.array([
                0, 3.9e-4, 4.7e-4, 1.9e-4, 9e-5, 2e-5
            ])).all())
        self.assertTrue(np.equal(
            np.stack(all_q_out).sum(axis = 0).round(5).ravel(), np.array([
                3.9e-4, 4.7e-4, 1.9e-4, 9e-5, 2e-5, -1e-5
            ])).all())
        # With higher influx
        all_q_in = []
        all_q_out = []
        for t, _ in enumerate(time_steps):
            if t == 0:
                vwc = self._sm_profile
            d_vwc, flows, _ = model.solve_vwc(1e-5, vwc, self._temp_profile, dt = dt)
            q_in, q_out = flows
            all_q_in.append(q_in)
            all_q_out.append(q_out)
            vwc = np.add(vwc, d_vwc)
        self.assertTrue(np.equal(
            np.stack(all_q_in).sum(axis = 0).round(5).ravel(), np.array([
                1.72e-3, 9.5e-4, 5.9e-4, 2e-4, 9e-5, 2e-5
            ])).all())
        self.assertTrue(np.equal(
            np.stack(all_q_out).sum(axis = 0).round(5).ravel(), np.array([
                9.5e-4, 5.9e-4, 2e-4, 9e-5, 2e-5, -1e-5
            ])).all())


if __name__ == '__main__':
    unittest.main()
