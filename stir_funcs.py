### This file contains functions to create stir objects

from stir import (ProjMatrixByBinSPECTUB, ProjectorByBinPairUsingProjMatrixByBin,
                   PoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat,
                   OSMAPOSLReconstruction3DFloat)

def make_projector(templ_sino, im, keep_views_in_cache=False):
    acq_model_matrix = ProjMatrixByBinSPECTUB() # create a SPECT porjection matrix object
    acq_model_matrix.set_keep_all_views_in_cache(keep_views_in_cache) # This keeps views in memory for a speed improvement
    acq_model_matrix.set_resolution_model(0.1,0.1) # Set a resolution model (just a guess!)
    projector = ProjectorByBinPairUsingProjMatrixByBin(acq_model_matrix)
    projector.set_up(templ_sino.get_proj_data_info(), im)
    return projector

def make_objective_function(templ_sino, projector):
    obj_function = PoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat()
    obj_function.set_proj_data_sptr(templ_sino)
    obj_function.set_projector_pair_sptr(projector)
    obj_function.set_recompute_sensitivity(False)
    return obj_function

def make_reconstructor(im, obj_function, subsets=7, subiters=7):
    recon = OSMAPOSLReconstruction3DFloat()
    recon.set_objective_function(obj_function)
    recon.set_num_subsets(subsets)
    recon.set_num_subiterations(subiters)
    recon.set_disable_output(True)
    recon.set_up(im)
    return recon