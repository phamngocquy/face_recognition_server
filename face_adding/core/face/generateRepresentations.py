import os
from subprocess import Popen, PIPE

openfaceDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pathAlignedImage = os.path.join(os.path.expanduser('~'), 'upload', 'aligned-images')
batchRepresent = os.path.join(openfaceDir, 'batch-represent', 'main.lua')


def batch_represent():
    pathGenerateRep = os.path.join(os.path.expanduser('~'), 'upload', 'generated-embeddings')
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


if __name__ == '__main__':
    batch_represent()
