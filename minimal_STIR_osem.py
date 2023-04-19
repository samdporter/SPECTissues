#%%
import stir
from sirf.STIR import MessageRedirector

from stir_funcs import (make_objective_function, make_projector,
                        make_reconstructor)

keep_views_in_cache = True # 

templ_sino_1 = stir.ProjData.read_from_file("template_sinogram.hs") # create an empty sinogram using a template
templ_sino_2 = stir.ProjData.read_from_file("template_sinogram.hs") # create an empty sinogram using a template

im = stir.FloatVoxelsOnCartesianGrid.read_from_file("init.hv")

target_1 = im.get_empty_copy()
target_1.fill(1.0)
target_2 = im.get_empty_copy()
target_2.fill(1.0)

projector_1 = make_projector(templ_sino_1, im)
projector_2 = make_projector(templ_sino_2, im)

projdata_1 = stir.ProjDataInMemory(templ_sino_1.get_exam_info(), templ_sino_1.get_proj_data_info())
projdata_2 = stir.ProjDataInMemory(templ_sino_2.get_exam_info(), templ_sino_2.get_proj_data_info())

projector_1.get_forward_projector().forward_project(projdata_1, im)
projector_2.get_forward_projector().forward_project(projdata_2, im)

obj_function_1 = make_objective_function(projdata_1, projector_1)
obj_function_2 = make_objective_function(projdata_2, projector_2)

recon_1 = make_reconstructor(target_1, obj_function_1, subsets=7, subiters=7)
recon_2 = make_reconstructor(target_2, obj_function_2, subsets=7, subiters=7)

recon_1.reconstruct(target_1)
recon_2.reconstruct(target_2)

target_1.write_to_file("recon_1_stir.hv")
target_2.write_to_file("recon_2_stir.hv")