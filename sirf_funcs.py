### This file contains functions to make the acquisition model, objective function and reconstructor objects

from sirf.STIR import (SPECTUBMatrix, AcquisitionModelUsingMatrix, 
                       make_Poisson_loglikelihood, OSMAPOSLReconstructor, 
                       EllipticCylinder)

def make_acquisition_model(templ_sino, im, keep_views_in_cache=False):
    # set up projection matrix objecy
    acq_model_matrix = SPECTUBMatrix()
    acq_model_matrix.set_keep_all_views_in_cache(keep_views_in_cache) # choose whethe to keep views in cache
    acq_model_matrix.set_resolution_model(0,0,full_3D=False)

    # make acquisiton model using projection matrix
    acq_model = AcquisitionModelUsingMatrix(acq_model_matrix)
    acq_model.set_up(templ_sino, im)
    return acq_model

def make_objective_function(templ_sino, acq_model, im):
    # objective function
    obj_fun = make_Poisson_loglikelihood(templ_sino)
    obj_fun.set_acquisition_model(acq_model)
    return obj_fun

def make_reconstructor(obj_fun, im, subsets=7, subiters=7):
    # reconstructor
    recon = OSMAPOSLReconstructor()
    recon.set_objective_function(obj_fun)
    recon.set_num_subsets(subsets)
    recon.set_num_subiterations(subiters)
    recon.set_up(im)
    return recon

def make_shape(im):
    shape = EllipticCylinder()
    shape.set_length(400)
    shape.set_radii((20,20))
    shape.set_origin((0, 20, -10))
    im.add_shape(shape, scale = 1)
    shape.set_radii((70,60))
    shape.set_origin((0, 0, 0))
    im.add_shape(shape, scale = 0.5)
    return im