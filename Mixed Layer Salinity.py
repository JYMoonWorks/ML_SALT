# Library 
import xarray as xr
import numpy as np

# Variables
# mld [ensemble, time, lat, lon] : mixed layer depth (m)
# salt [ensemble, time, depth, lat, lon] : salinity (psu)

# Depth weights (dz from LOVECLIM)
depth_weight = np.array([747.64, 734.28, 713.94, 681.78, 630.00, 548.92, 436.66, 313.18, 209.44, 138.18, 93.38, 65.48, 47.68, 35.84, 27.70, 21.88, 17.64, 14.42, 11.96, 10.00])
depth_weight = xr.DataArray(depth_weight, dims=['depth'], coords={'depth':test_ocean.depth})  # test ocean file contains depth information.
depth_weight_broadcasted = depth_weight.broadcast_like(salt) 

# Function
def ML_salt(mld, salt):
    mld_broadcasted = mld.broadcast_like(salt)
    salt_masked = salt.where( salt.depth < mld_broadcasted )
    salt_weighted = (salt_masked * depth_weight_broadcasted).sum(dim='depth')
    # Since salinity values in the mixed layer depth could be NaN, salt.notnull() is included.
    mld_sum = depth_weight_broadcasted.where( (salt.depth < mld_broadcasted) & (salt.notnull()) ).sum(dim='depth', skipna=True)   
    ML_salt = salt_weighted / mld_sum

    return ML_salt
