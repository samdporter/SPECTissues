### test sirf an stir reconstructions are the same

import glob
import os
import subprocess

from sirf.STIR import ImageData

#remove old files
for f in glob.glob("recon*"):
    os.remove(f)

subprocess.run(["python3", "minimal_SIRF_osem.py"])
subprocess.run(["python3", "minimal_STIR_osem.py"])

def test_sirf():
    im_1 = ImageData("recon_1_sirf.hv")
    im_2 = ImageData("recon_2_sirf.hv")
    assert im_1.as_array().all() == im_2.as_array().all()

def test_stir():
    im_1 = ImageData("recon_1_stir.hv")
    im_2 = ImageData("recon_2_stir.hv")
    assert im_1.as_array().all() == im_2.as_array().all()

def test_recon1():
    im_sirf = ImageData("recon_1_sirf.hv")
    im_stir = ImageData("recon_1_stir.hv")
    assert im_sirf.as_array().all() == im_stir.as_array().all()

def test_recon2():
    im_sirf = ImageData("recon_2_sirf.hv")
    im_stir = ImageData("recon_2_stir.hv")
    assert im_sirf.as_array().all() == im_stir.as_array().all()

for f in glob.glob("tmp_*"):
    os.remove(f)