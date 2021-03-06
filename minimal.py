import sirf.STIR as STIR
import brainweb
from tqdm.auto import tqdm
import numpy
from pathlib import Path
import os
import psutil

fname, url= sorted(brainweb.utils.LINKS.items())[0]
files = brainweb.get_file(fname, url, ".")
data = brainweb.load_file(fname)

path = Path(__file__).resolve().parent

dir = os.path.dirname(__file__)

msg = STIR.MessageRedirector(dir+"/info.txt", dir+"/warnings.txt", dir+"/error.txt")

brainweb.seed(1337)

for f in tqdm([fname], desc="mMR ground truths", unit="subject"):
    vol = brainweb.get_mmr_fromfile(f, petNoise=1, t1Noise=0.75, t2Noise=0.75, petSigma=1, t1Sigma=1, t2Sigma=1)
fdg_arr = vol['PET']

# Select central slice
central_slice = fdg_arr.shape[0]//2
fdg_arr = fdg_arr[central_slice, :, :]

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
am = STIR.AcquisitionModelUsingMatrix(acq_model_matrix)
am.set_up(templ_sino, fdg)

sino = am.forward(fdg)
bp = am.backward(sino)
sino = am.forward(bp)
bp = am.backward(sino)

print("all done with individuals")

sino = am.forward(fdg)
for i in range (100):
    bp = am.backward(sino)
    sino = am.forward(bp)
    print('The CPU usage is: ', psutil.cpu_percent(4))
    print('RAM memory % used:', psutil.virtual_memory()[2])

