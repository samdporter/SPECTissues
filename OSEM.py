import sirf.STIR as STIR
import brainweb
from tqdm.auto import tqdm
import numpy
from pathlib import Path
import os
import psutil

def main():

    fname, url= sorted(brainweb.utils.LINKS.items())[0]
    files = brainweb.get_file(fname, url, ".")
    data = brainweb.load_file(fname)

    path = Path(__file__).resolve().parent

    dir = os.path.dirname(__file__)

    msg = STIR.MessageRedirector(dir+"/info.txt", dir+"/warnings.txt", dir+"/error.txt")

    brainweb.seed(1337)

    for f in tqdm([fname], desc="mMR ground truths", unit="subject"):
            vol_amyl = brainweb.get_mmr_fromfile(f, petNoise=1, t1Noise=0.75, t2Noise=0.75,petSigma=1, t1Sigma=1, t2Sigma=1,PetClass=brainweb.Amyloid)
            vol = brainweb.get_mmr_fromfile(f, petNoise=1, t1Noise=0.75, t2Noise=0.75, petSigma=1, t1Sigma=1, t2Sigma=1)
    amyl_arr = vol_amyl['PET']
    fdg_arr = vol['PET']

    # Select central slice
    central_slice = fdg_arr.shape[0]//2
    fdg_arr = fdg_arr[central_slice, :, :]
    amyl_arr = amyl_arr[central_slice, :, :]

    # Select a central ROI with 120x120
    idim = [120,120]
    offset = (numpy.array(fdg_arr.shape) - numpy.array(idim)) // 2
    fdg_arr = fdg_arr[offset[0]:offset[0]+idim[0], offset[1]:offset[1]+idim[1]]
    amyl_arr = amyl_arr[offset[0]:offset[0]+idim[0], offset[1]:offset[1]+idim[1]]

    # Now we make sure our image is of shape (1, 120, 120) 
    fdg_arr = fdg_arr[numpy.newaxis,...] 
    amyl_arr = amyl_arr[numpy.newaxis,...]

    templ_sino = STIR.AcquisitionData(os.path.join(path,"template_sinogram.hs")) # create an empty sinogram using a template

    im = STIR.ImageData(templ_sino)
    dim = fdg_arr.shape
    voxel_size=im.voxel_sizes()
    im.initialise(dim,(voxel_size[0]*2, voxel_size[1], voxel_size[2]))
    fdg = im.clone().fill(fdg_arr)
    amyl = im.clone().fill(amyl_arr)

    acq_model_matrix0 = STIR.SPECTUBMatrix()
    acq_model_matrix0.set_resolution_model(0,0,full_3D=False)
    am0 = STIR.AcquisitionModelUsingMatrix(acq_model_matrix0)
    am0.set_up(templ_sino, fdg)

    acq_model_matrix1 = STIR.SPECTUBMatrix()
    acq_model_matrix1.set_resolution_model(0,0,full_3D=False)
    am1 = STIR.AcquisitionModelUsingMatrix(acq_model_matrix1)
    am1.set_up(templ_sino, amyl)

    obj_fun0 = STIR.make_Poisson_loglikelihood(templ_sino)
    obj_fun0.set_acquisition_model(am0)
    obj_fun1 = STIR.make_Poisson_loglikelihood(templ_sino)
    obj_fun1.set_acquisition_model(am1)

    reconstructor0 = STIR.OSMAPOSLReconstructor()
    reconstructor0.set_objective_function(obj_fun0)
    reconstructor0.set_num_subsets(21)
    reconstructor0.set_num_subiterations(42)
    reconstructor0.set_up(fdg)
    reconstructor1 = STIR.OSMAPOSLReconstructor()
    reconstructor1.set_objective_function(obj_fun0)
    reconstructor1.set_num_subsets(21)
    reconstructor1.set_num_subiterations(42)
    reconstructor1.set_up(amyl)

    reconstructor0.reconstruct(fdg)
    reconstructor0.reconstruct(amyl)

    fdg.show()
    amyl.show()

main()
print("done")