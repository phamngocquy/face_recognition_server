import os


class Config:
    dirPath = os.path.expanduser('~')
    storePath = os.path.join('training-images', dirPath)
