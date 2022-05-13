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
uMap_arr = uMap_arr[offset[0]:offset[0]+idim[0], offset[1]:offset[1]+idim[1]]

# Now we make sure our image is of shape (1, 120, 120) 
fdg_arr = fdg_arr[numpy.newaxis,...]
uMap_arr = uMap_arr[numpy.newaxis,...]

templ_sino = STIR.AcquisitionData(os.path.join(path,"template_sinogram.hs")) # create an empty sinogram using a template

im = STIR.ImageData(templ_sino)
dim = fdg_arr.shape
voxel_size=im.voxel_sizes()
im.initialise(dim,(voxel_size[0]*2, voxel_size[1], voxel_size[2]))
fdg = im.clone().fill(fdg_arr)
uMap = im.clone().fill(uMap_arr)

acq_model_matrix = STIR.SPECTUBMatrix()
acq_model_matrix.set_resolution_model(0,0,full_3D=False )
acq_model_matrix.set_attenuation_image(uMap)
am = STIR.AcquisitionModelUsingMatrix(acq_model_matrix)
am.set_up(templ_sino, fdg)

sino = am.forward(fdg)
bp = am.backward(sino)
sino = am.forward(bp)
bp = am.backward(sino)

sino_arr = am.forward(fdg).as_array()
noisy_arr = np.random.poisson(sino_arr/10)*10
noisy = sino.clone().fill(noisy_arr)

print("all done with individuals")

print("testing loop")
try:
    sino = am.forward(fdg)
    for i in range (10):
        bp = am.backward(sino)
        sino = am.forward(bp)
    print("loop OK")
except:
    print("loop failed")

print("testing OSEM")
try:
    obj_fun = STIR.make_Poisson_loglikelihood(sino)
    obj_fun.set_acquisition_model(am)

    reconstructor = STIR.OSMAPOSLReconstructor()
    reconstructor.set_objective_function(obj_fun)
    reconstructor.set_num_subsets(21)
    reconstructor.set_num_subiterations(8)
    reconstructor.set_up(bp)

    reconstructor.reconstruct(bp)
    print("OSEM successful")
except:
    print("OSEM failed")

normA = am.norm()
tau=1/normA
sigma=1/normA

## using numpy backend to avoid segmentation fault ##
grad = operators.GradientOperator(fdg, backend = 'numpy')
R = functions.MixedL21Norm()
KL = functions.KullbackLeibler(b=noisy,epsilon = 0.0001)
g = functions.IndicatorBox(lower=0)

F = functions.BlockFunction(KL,R)
K = operators.BlockOperator(am,grad)

algorithm = algorithms.LADMM(initial = bp.get_uniform_copy(0.5), f = g, g=F, operator = K, sigma = sigma, tau = tau, max_iteration=100,update_objective_interval=1, use_axpby=False) 

## fails here if using a c-backend for grad ##
algorithm.run(verbose=2)
out = algorithm.solution





