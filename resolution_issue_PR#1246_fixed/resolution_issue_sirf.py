#%%
from sirf.STIR import *

AcquisitionData.set_storage_scheme('memory')

import os
data_path = "/home/sam/working/SPECT_issues/test_data"

keep_views_in_cache = True # 

msg = MessageRedirector("info.txt", "warnings.txt", "error.txt")

templ_sino = AcquisitionData(os.path.join(data_path, "peak_1_projdata__f1g1d0b0.hs"))

# create an empty image using the template data
im = ImageData(os.path.join(data_path, "NEMA_SPECT_template.hv"))

#%%
acq_model_matrix = SPECTUBMatrix()
acq_model_matrix.set_attenuation_image(im)
acq_model_matrix.set_keep_all_views_in_cache(keep_views_in_cache)
acq_model_matrix.set_resolution_model(0.93, 0.03, False)

# make acquisiton model using projection matrix
acq_model = AcquisitionModelUsingMatrix(acq_model_matrix)

#%%
acq_model.set_up(templ_sino, im)
# %%