#%%
import sirf.STIR as STIR

keep_views_in_cache = False # 

msg = STIR.MessageRedirector("info.txt", "warnings.txt", "error.txt")

def make_shape(im):
    shape = STIR.EllipticCylinder()
    shape.set_length(400)
    shape.set_radii((20,20))
    shape.set_origin((0, 20, -10))
    im.add_shape(shape, scale = 1)
    shape.set_radii((70,60))
    shape.set_origin((0, 0, 0))
    im.add_shape(shape, scale = 0.5)

templ_sino_1 = STIR.AcquisitionData("template_sinogram.hs") # create an empty sinogram using a template
templ_sino_2 = STIR.AcquisitionData("template_sinogram.hs") # create an empty sinogram using a template

# create an empty image using the template data
im_1 = templ_sino_1.create_uniform_image(0)
im_1 = im_1.zoom_image(zooms = (0.5,1,1)) #zoom because SPECT is 360 degrees (based on PET)

im_2 = templ_sino_2.create_uniform_image(0)
im_2 = im_2.zoom_image(zooms = (0.5,1,1)) #zoom because SPECT is 360 degrees (based on PET)

# create some shapes and add to the image 
make_shape(im_1)
make_shape(im_2)

#%%
# set up projection matrix objecy
acq_model_matrix_1 = STIR.SPECTUBMatrix()
acq_model_matrix_1.set_keep_all_views_in_cache(keep_views_in_cache) # choose whethe to keep views in cache
acq_model_matrix_1.set_resolution_model(0,0,full_3D=False)

#%%
acq_model_matrix_2 = STIR.SPECTUBMatrix()
acq_model_matrix_2.set_keep_all_views_in_cache(keep_views_in_cache) # choose whethe to keep views in cache
acq_model_matrix_2.set_resolution_model(0,0,full_3D=False)

#%%
# create acquisiton model using projection matrix
acq_model_1 = STIR.AcquisitionModelUsingMatrix(acq_model_matrix_1)
acq_model_1.set_up(templ_sino_1, im_1)

#%%
acq_model_2 = STIR.AcquisitionModelUsingMatrix(acq_model_matrix_2)
acq_model_2.set_up(templ_sino_2, im_2)
