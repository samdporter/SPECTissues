#%%
from stir import *
from stirextra import to_numpy
import numpy as np

import os
path = os.path.dirname(os.path.realpath(__file__))

keep_views_in_cache = True

templ_sino = ProjData.read_from_file(os.path.join(path, "test_data/peak_1_projdata__f1g1d0b0.hs")) # create an empty sinogram using a template

im = FloatVoxelsOnCartesianGrid.read_from_file(os.path.join(path, "test_data/NEMA_SPECT_template.hv"))

target = im.get_empty_copy()
target.fill(1.0)

#%% create an acquisition model using a projection matrix
acq_model_matrix = ProjMatrixByBinSPECTUB() # create a SPECT porjection matrix object
acq_model_matrix.set_keep_all_views_in_cache(keep_views_in_cache) # This keeps views in memory for a speed improvement
acq_model_matrix.set_resolution_model(0.93, 0.03, False) # Set a resolution model (just a guess!)
projector = ProjectorByBinPairUsingProjMatrixByBin(acq_model_matrix)
#%% set up the projector
projector.set_up(templ_sino.get_proj_data_info(), target)

#%% forward project
projdata = ProjDataInMemory(templ_sino.get_exam_info(), templ_sino.get_proj_data_info())
projector.get_forward_projector().forward_project(projdata, target)

# not zero as only called set_up once
call_1 = np.sum(to_numpy(projdata))

#%% make sure it's not an issue with wiping the target image or the projdata
target = im.get_empty_copy()
target.fill(1.0)

projdata = ProjDataInMemory(templ_sino.get_exam_info(), templ_sino.get_proj_data_info())

#%% set up the projector again
projector.set_up(templ_sino.get_proj_data_info(), target)

#%% forward project again
projector.get_forward_projector().forward_project(projdata, target)

# should be zero if problem is still present
call_2 = np.sum(to_numpy(projdata))

print(f"call 1 sum: {call_1}")
print(f"call 2 sum: {call_2}")
