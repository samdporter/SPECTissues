import sirf.STIR as STIR
import brainweb
from tqdm.auto import tqdm
import numpy
import os

def main():

    dir = os.path.dirname(__file__) # path to parent directory

    msg = STIR.MessageRedirector(dir+"/info.txt", dir+"/warnings.txt", dir+"/error.txt") # redirtect messages

    # get brainweb data
    fname, url= sorted(brainweb.utils.LINKS.items())[0]
    brainweb.seed(1337)
    for f in tqdm([fname], desc="mMR ground truths", unit="subject"):
            vol_amyl = brainweb.get_mmr_fromfile(f, petNoise=1, t1Noise=0.75, t2Noise=0.75,petSigma=1, t1Sigma=1, t2Sigma=1,PetClass=brainweb.Amyloid)
            vol = brainweb.get_mmr_fromfile(f, petNoise=1, t1Noise=0.75, t2Noise=0.75, petSigma=1, t1Sigma=1, t2Sigma=1)
    amyl_arr = vol_amyl['PET']
    fdg_arr = vol['PET']

    # Select central slice for 2D array
    central_slice = fdg_arr.shape[0]//2
    fdg_arr = fdg_arr[central_slice, :, :]
    amyl_arr = amyl_arr[central_slice, :, :]

    # Select a central ROI with 120x120 pixels
    idim = [120,120]
    offset = (numpy.array(fdg_arr.shape) - numpy.array(idim)) // 2
    fdg_arr = fdg_arr[offset[0]:offset[0]+idim[0], offset[1]:offset[1]+idim[1]]
    amyl_arr = amyl_arr[offset[0]:offset[0]+idim[0], offset[1]:offset[1]+idim[1]]

    # make sure our image is of shape (1, 120, 120) 
    fdg_arr = fdg_arr[numpy.newaxis,...] 
    amyl_arr = amyl_arr[numpy.newaxis,...]

    templ_sino = STIR.AcquisitionData(os.path.join(dir,"template_sinogram.hs")) # create an empty sinogram using a template

    # initialise 1x120x120 imge using template sino
    im = STIR.ImageData(templ_sino)
    dim = fdg_arr.shape
    voxel_size=im.voxel_sizes()
    im.initialise(dim,(voxel_size[0]*2, voxel_size[1], voxel_size[2]))
    fdg = im.clone().fill(fdg_arr)
    amyl = im.clone().fill(amyl_arr)

    # create SPECT acquisition modeals
    acq_model_matrix0 = STIR.SPECTUBMatrix()
    acq_model_matrix0.set_resolution_model(0,0,full_3D=False)
    am0 = STIR.AcquisitionModelUsingMatrix(acq_model_matrix0)
    am0.set_up(templ_sino, fdg)

    acq_model_matrix1 = STIR.SPECTUBMatrix()
    acq_model_matrix1.set_resolution_model(0,0,full_3D=False)
    am1 = STIR.AcquisitionModelUsingMatrix(acq_model_matrix1)
    am1.set_up(templ_sino, amyl)
    print("acquisition models set up")

    # create objective functions
    obj_fun0 = STIR.make_Poisson_loglikelihood(templ_sino)
    obj_fun0.set_acquisition_model(am0)
    obj_fun1 = STIR.make_Poisson_loglikelihood(templ_sino)
    obj_fun1.set_acquisition_model(am1)
    print("objective functions ready")

    # create reconstructors
    reconstructor0 = STIR.OSMAPOSLReconstructor()
    reconstructor0.set_objective_function(obj_fun0)
    reconstructor0.set_num_subsets(21)
    reconstructor0.set_num_subiterations(42)
    reconstructor0.set_up(fdg)
    print("reconstructor 0 set up")
    reconstructor1 = STIR.OSMAPOSLReconstructor()
    reconstructor1.set_objective_function(obj_fun0)
    reconstructor1.set_num_subsets(21)
    reconstructor1.set_num_subiterations(42)
    reconstructor1.set_up(amyl)
    print("reconstructor 1 set up")

    reconstructor0.reconstruct(fdg)
    print("fdg reconstructed")
    reconstructor0.reconstruct(amyl)
    print("amyl reconstructed")

    fdg.show()
    amyl.show()

main()
print("done")