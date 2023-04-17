#%%
import stir

keep_views_in_cache = False # 

templ_sino_1 = stir.ProjData.read_from_file("template_sinogram.hs") # create an empty sinogram using a template
templ_sino_2 = stir.ProjData.read_from_file("template_sinogram.hs") # create an empty sinogram using a template

im = stir.FloatVoxelsOnCartesianGrid.read_from_file("init.hv")

#%%
# set up projection matrix objecy
acq_model_matrix_1 = stir.ProjMatrixByBinSPECTUB() # create a SPECT porjection matrix object
acq_model_matrix_1.set_keep_all_views_in_cache(False) # This keeps views in memory for a speed improvement
acq_model_matrix_1.set_resolution_model(0.1,0.1) # Set a resolution model (just a guess!)

#%%
acq_model_matrix_2 = stir.ProjMatrixByBinSPECTUB() # create a SPECT porjection matrix object
acq_model_matrix_2.set_keep_all_views_in_cache(False) # This keeps views in memory for a speed improvement
acq_model_matrix_2.set_resolution_model(0.1,0.1) # Set a resolution model (just a guess!)

#%%
# create acquisiton model using projection matrix
projector_1 = stir.ProjectorByBinPairUsingProjMatrixByBin(acq_model_matrix_1)
projector_1.set_up(templ_sino_1.get_proj_data_info(), im)

#%%
projector_2 = stir.ProjectorByBinPairUsingProjMatrixByBin(acq_model_matrix_2)
projector_2.set_up(templ_sino_2.get_proj_data_info(), im)