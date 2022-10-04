import subprocess
subprocess.run('source /usr/.bashrc', shell=True)

import os
import sirf.STIR as STIR
import cil.optimisation.algorithms as algorithms
import cil.optimisation.functions as functions
import cil.optimisation.operators as operators
from cil.optimisation.algorithms.ADMM import LADMM
import brainweb
from tqdm.auto import tqdm
import numpy
from pathlib import Path
import numpy as np
import pdb
pdb.set_trace()
STIR.set_default_num_omp_threads()

dir = os.path.dirname(__file__)

msg = STIR.MessageRedirector(dir+"/info.txt", dir+"/warnings.txt", dir+"/error.txt")

fname, url= sorted(brainweb.utils.LINKS.items())[0]
files = brainweb.get_file(fname, url, ".")
data = brainweb.load_file(fname)

path = Path(__file__).resolve().parent

brainweb.seed(1337)

for f in tqdm([fname], desc="mMR ground truths", unit="subject"):
    vol = brainweb.get_mmr_fromfile(f, petNoise=1, t1Noise=0.75, t2Noise=0.75, petSigma=1, t1Sigma=1, t2Sigma=1)
fdg_arr = vol['PET']
uMap_arr = vol['uMap']

# Select central slice
central_slice = fdg_arr.shape[0]//2

fdg_arr = fdg_arr[central_slice, :, :]
uMap_arr = uMap_arr[central_slice, :, :]

# Select a central ROI with 120x120
idim = [120,120]
offset = (numpy.array(fdg_arr.shape) - numpy.array(idim)) // 2
fdg_arr = fdg_arr[offset[0]:offset[0]+idim[0], offset[1]:offset[1]+idim[1]]

# Now we make sure our image is of shape (1, 120, 120) 
fdg_arr = fdg_arr[numpy.newaxis,...]

templ_sino = STIR.AcquisitionData(os.path.join(path,"template_sinogram.hs")) # create an empty sinogram using a template

im = STIR.ImageData(templ_sino)
dim = fdg_arr.shape
voxel_size=im.voxel_sizes()
im.initialise(dim,(voxel_size[0]*2, voxel_size[1], voxel_size[2]))
fdg = im.clone().fill(fdg_arr)

acq_model_matrix = STIR.SPECTUBMatrix()
acq_model_matrix.set_resolution_model(0,0,full_3D=False)
acq_model_matrix.set_keep_all_views_in_cache(False)
print(acq_model_matrix.set_keep_all_views_in_cache())
am = STIR.AcquisitionModelUsingMatrix(acq_model_matrix)
am.set_up(templ_sino, fdg)

obj_fun = STIR.make_Poisson_loglikelihood(templ_sino)
obj_fun.set_acquisition_model(am)

reconstructor = STIR.OSMAPOSLReconstructor()
reconstructor.set_objective_function(obj_fun)
reconstructor.set_num_subsets(21)
reconstructor.set_num_subiterations(8)
reconstructor.set_up(fdg)

reconstructor.reconstruct(fdg)

#fdg.show()