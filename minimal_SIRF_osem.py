#%%
import os

from sirf.STIR import MessageRedirector, AcquisitionData

AcquisitionData.set_storage_scheme('memory')

from sirf_funcs import (make_acquisition_model, make_objective_function,
                        make_reconstructor, make_shape)

keep_views_in_cache = True # 

msg = MessageRedirector("info.txt", "warnings.txt", "error.txt")

templ_sino_1 = AcquisitionData("template_sinogram.hs") # create an empty sinogram using a template
templ_sino_2 = AcquisitionData("template_sinogram.hs") # create an empty sinogram using a template

# create an empty image using the template data
im_1 = templ_sino_1.create_uniform_image(0)
im_1 = im_1.zoom_image(zooms = (0.5,1,1)) #zoom because SPECT is 360 degrees (based on PET)

im_2 = templ_sino_2.create_uniform_image(0)
im_2 = im_2.zoom_image(zooms = (0.5,1,1)) #zoom because SPECT is 360 degrees (based on PET)

# create some shapes and add to the image
im_1 = make_shape(im_1)
im_2 = make_shape(im_2)

if not os.path.exists('init.hv'):
    im_1.write('init.hv')

#%%
acq_model_1 = make_acquisition_model(templ_sino_1, im_1, 
                                     keep_views_in_cache=keep_views_in_cache)
acq_model_2 = make_acquisition_model(templ_sino_2, im_2, 
                                     keep_views_in_cache=keep_views_in_cache)

# forward project to simulate noiseless data
sino_1 = acq_model_1.forward(im_1)
sino_2 = acq_model_2.forward(im_2)

#%%
obj_fun_1 = make_objective_function(sino_1, acq_model_1, im_1)
obj_fun_2 = make_objective_function(sino_2, acq_model_2, im_2)

#%%
recon_1 = make_reconstructor(obj_fun_1, im_1, subsets=7, subiters=7)
recon_2 = make_reconstructor(obj_fun_2, im_2, subsets=7, subiters=7)

#%%
init_image_1 = im_1.get_uniform_copy(1)
init_image_2 = im_2.get_uniform_copy(1)

recon_1.set_current_estimate(init_image_1)
recon_2.set_current_estimate(init_image_2)

#%%
# reconstruction
recon_1.reconstruct(init_image_1)
recon_2.reconstruct(init_image_2)

init_image_1.write("recon_1_sirf.hv")
init_image_2.write("recon_2_sirf.hv")
# %%
