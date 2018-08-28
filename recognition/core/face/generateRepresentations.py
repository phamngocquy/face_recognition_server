from subprocess import Popen, PIPE
from face.alignImage import *
from face.config.config import Config
import shutil


def batch_represent():
    cmd = [Config.batchRepresent, "-data",
           Config.alignedImagePath,
           "-outDir",
           Config.repsImagePath, '-cuda', '-cache']
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    (out, err) = p.communicate()
    print(out)
    print(err)


def clean():
    if os.path.exists(Config.alignedCache):
        os.remove(Config.alignedCache)
        print("=>>> Clean cache align images.")
    if os.path.exists(Config.repsImagePath):
        shutil.rmtree(Config.repsImagePath)
        print("=>>> Delete old model.")


def doGeneratePres():
    clean()
    doAlignImage()
    batch_represent()
