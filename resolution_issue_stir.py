#%%
from stir import *

import os
path = os.path.dirname(os.path.realpath(__file__))

keep_views_in_cache = True # 

templ_sino = ProjData.read_from_file(os.path.join(path, "test_data/peak_1_projdata__f1g1d0b0.hs")) # create an empty sinogram using a template

im = FloatVoxelsOnCartesianGrid.read_from_file(os.path.join(path, "test_data/NEMA_SPECT_template.hv"))

target = im.get_empty_copy()
target.fill(1.0)

#%%
acq_model_matrix = ProjMatrixByBinSPECTUB() # create a SPECT porjection matrix object
acq_model_matrix.set_keep_all_views_in_cache(keep_views_in_cache) # This keeps views in memory for a speed improvement
acq_model_matrix.set_resolution_model(0.93, 0.03, False) # Set a resolution model (just a guess!)
projector = ProjectorByBinPairUsingProjMatrixByBin(acq_model_matrix)

#%%
obj_function = PoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat()
obj_function.set_proj_data_sptr(templ_sino)
obj_function.set_projector_pair_sptr(projector)
# %%
obj_function.set_up(target)