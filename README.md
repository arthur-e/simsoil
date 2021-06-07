(Very) Simple Soil Hydrology Model
==================================

[![DOI](https://zenodo.org/badge/374736375.svg)](https://zenodo.org/badge/latestdoi/374736375)

A very simple, point-scale soil hydrology model based on the modified Richards equation from [the Community Land Model (CLM)](https://escomp.github.io/ctsm-docs/versions/release-clm5.0/html/tech_note/index.html), version 5.0, with some equations from CLM 4.x and pedotransfer functions from Balland et al. (2008). The basic model characterizes the vertical movement of water in a soil column and lateral sub-surface drainage at sub-daily time steps, driven by:

- Initial soil volumetric water content (VWC) state;
- Mean daily infiltration rate (kg m-2 sec-1);
- Mean daily potential transpiration (kg m-2 sec-1);
- Mean soil temperature in each layer;
- Soil texture and porosity;
- Fraction of land area that is saturated;

The soil column is considered to be at a point location; i.e., there is no connectivity between lateral sub-surface runoff and any other soil column. The approach here is designed to produce reasonable results for a wide variety of soils. The soil hydrology model has been tested using soil infiltration rates, soil texture, and soil temperature data from [the SMAP Level 4 Soil Moisture (L4SM) product](https://nsidc.org/data/SPL4SMGP) and the Catchment land model (Koster et al. 2000).

**Significance:**

This is a small library that enables the generation of reasonable soil moisture profile time series data at point-scale. With this library, provided you have reasonable surface meteorology driver data, it is not necessary to install, configure, and spin-up a more sophisticated terrestrial biosphere model or land model. Your feedback is welcome! Get in touch with me or submit a Pull Request.

If this library doesn't suit your needs, you might check out these related Python libraries:

- [`swb` (soil water balance)](https://swb.readthedocs.io/en/latest/index.html)


Dependencies
------------

- Python 3.5+
- `scipy` 1.3+
- `numpy` 1.13.3+
- `cached_property` 1.5.1+


Installation
------------

It is recommended that you install within either a `virtualenv` or `conda` virtual environment.

```sh
$ python setup.py install .
```

**To install using `pip`:**

```sh
$ pip install .
```

To install `simsoil` in "development mode," which enables you to edit the source code:

```sh
$ pip install -e .
```

**Tests can be run by:**

```sh
$ python simsoil/tests/tests.py
```

Note that the test suite may take up to a minute to complete.


Documentation
-------------

`pdoc3` is required for generating the documentation. It can be installed using `pip` with:

```sh
$ pip install simsoil[docs]
```

To build the HTML documentation:

```sh
pdoc --html -c latex_math=True --force -o <output_dir> <simsoil_python_dir>
```


Getting Started
---------------

It's generally necessary to spin-up each soil column to a quasi-equilibrium state, beginning with an arbitrary guess at the initial soil VWC state, e.g., `0.15 m3 m-3`. Potential transpiration (PET) can be calculated using the `simsoil.transpiration` module or taken from MODIS MOD16.

```py
from simsoil.core import InfiltrationModel, SoilModel, DEPTHS, SOC_RATIOS

# Define the soil column
soil = SoilProfile(
  pft = 1, soc = SOC_RATIOS * 1e3, sand = 0.6, clay = 0.1,
  porosity = 0.4, bedrock = 3, slope = 0.01, depths_m = DEPTHS)

model = InfiltrationModel(soil_model = soil)

# e.g., spin-up for 20 years
n_days = 20 * 365

# e.g., Initial soil VWC in each layer = 0.15 m3 m-3
init_vwc = np.ones(DEPTHS.shape) * 0.15

# With adaptive = False, do not use adaptive time stepping; sub-daily
#   time steps will always be hourly (3600 seconds);
# With climatology = True, we can provide a 365-day climatology for
#   temperature, PET, and infiltration that will be recycled every year
results = model.run(
  init_vwc, temp_profile, pet, influx, f_sat, dt = 3600,
  n_days = n_days, climatology = True, adaptive = False)
```

Suggested driver datasets:

- Daily infiltration rate, e.g., from SMAP L4SM `Geophysical_Data/soil_water_infiltration_flux` field.
- Soil temperature in each layer, e.g., from SMAP L4SM `Geophysical_Data/soil_temp_layer*` fields (if using same soil layers as the Catchment model).
- Soil texture (sand, clay content) and porosity from SMAP L4SM `Land-Model-Constants_data`.
- Soil texture (sand, clay, and soil organic carbon content) from [SoilGrids 250m](https://soilgrids.org/).


Debugging
---------

In `InfiltrationModel.step_daily()`, it may be useful to add interactive debugging (after installing `ipdb`); for example, the following code block might be inserted anywhere after the call to `solve_vwc()`:

```py
# DEBUG: Tracking water flows can be helpful
if self._debug:
    vwc1 = np.add(vwc, x)
    vwc2 = vwc1 + runoff # Runoff is negative
    try:
        vwc3 = rebalance(vwc2, temp_profile, thickness_mm)
    except AssertionError:
        # i.e., rebalance() exceeded max. number of operations
        import ipdb
        ipdb.set_trace()
    if not np.logical_and(0 <= vwc3, vwc3 <= 1).all():
        # i.e., soil VWC exceeds physical limits
        import ipdb
        ipdb.set_trace()
```


Acknowledgments
---------------

Thanks to Colin Brust for providing the MOD16 source code for reference. Thanks also to Randy Koster and Rolf Reichle (NASA GMAO) for their feedback on an early version of this work.
