import os
import shutil
from subprocess import Popen, PIPE

openfaceDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pathAlignedImage = os.path.join(os.path.expanduser('~'), 'upload', 'aligned-images')
batchRepresent = os.path.join(openfaceDir, 'batch-represent', 'main.lua')
pathCacheAlign = os.path.join(pathAlignedImage, "cache.t7")
pathGenerateRep = os.path.join(os.path.expanduser('~'), 'upload', 'generated-embeddings')


def clean():
    if os.path.exists(pathCacheAlign):
        os.remove(pathCacheAlign)
        print("====> del align cache")
    if os.path.exists(pathGenerateRep):
        print("====> del generate reps")
        shutil.rmtree(pathGenerateRep)


def batch_represent():
    clean()
    if not os.path.exists(pathGenerateRep):
        os.makedirs(pathGenerateRep)

    cmd = [batchRepresent, "-data",
           pathAlignedImage,
           "-outDir",
           pathGenerateRep]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    (out, err) = p.communicate()
    print(out)
    print(err)
