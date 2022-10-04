import sirf.STIR as STIR

keep_views_in_cache = False # 

msg = STIR.MessageRedirector("info.txt", "warnings.txt", "error.txt")

templ_sino = STIR.AcquisitionData("template_sinogram.hs") # create an empty sinogram using a template

# create an empty image using the template data
im = templ_sino.create_uniform_image(0)
im = im.zoom_image(zooms = (0.5,1,1)) #zoom because SPECT is 360 degrees (based on PET)

# create some shapes and add to the image 
shape = STIR.EllipticCylinder()
shape.set_length(400)
shape.set_radii((20,20))
shape.set_origin((0, 20, -10))
im.add_shape(shape, scale = 1)
shape.set_radii((70,60))
shape.set_origin((0, 0, 0))
im.add_shape(shape, scale = 0.5)

# set up projection matrix objecy
acq_model_matrix = STIR.SPECTUBMatrix()
acq_model_matrix.set_resolution_model(0,0,full_3D=False)
acq_model_matrix.set_keep_all_views_in_cache(keep_views_in_cache) # choose whethe to keep views in cache

# create acquisiton model using projection matrix
am = STIR.AcquisitionModelUsingMatrix(acq_model_matrix)
am.set_up(templ_sino, im)

# forward project to simulate noiseless data
sino = am.forward(im)

# create poisson log likelihood data fidelity term
obj_fun = STIR.make_Poisson_loglikelihood(sino)
obj_fun.set_acquisition_model(am)

# set up reconstructor object
reconstructor = STIR.OSMAPOSLReconstructor()
reconstructor.set_objective_function(obj_fun)
reconstructor.set_num_subsets(21)
reconstructor.set_num_subiterations(21)
reconstructor.set_up(im)

# reconstruct and show image
result = im.get_uniform_copy(1)
reconstructor.reconstruct(result)
result.show()
