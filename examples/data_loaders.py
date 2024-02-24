### Data loaders for the irrigation mapping project

# Import the required modules
import netCDF4 as nc
import numpy as np

### Function to load the WRF training data
def load_WRFSCM_training_data(fpath):

    # Open the file and extract variables
    fn = nc.Dataset(fpath)
    vegfra = fn.variables['VEGFRA'][:] # 1-D
    irr = fn.variables['IRR'][:]        # 1-D (the target)
    sm = fn.variables['SM'][:]          # 1-D
    lst = fn.variables['LST'][:]        # 2-D (samples, features)
    time = fn.variables['SimTime'][:]   # 2-D (samples, features)
    
    # Normalize the variables
    vegfra = (vegfra-np.mean(vegfra))/np.std(vegfra)
    sm = (sm-np.mean(sm))/np.std(sm)
    lst = (lst-np.mean(lst, axis=0))/np.std(lst, axis=0)
    irr2 = (irr-np.mean(irr))/np.std(irr)
       
    # Stack into a single array
    X = np.stack([vegfra, sm]+list([lst[:,i] for i in range(lst.shape[1])]), axis=1).astype('double')
    y = irr2
    y_scales = (np.mean(irr), np.std(irr))
    
    # Return the training inputs and the targets
    return X, y, y_scales
    