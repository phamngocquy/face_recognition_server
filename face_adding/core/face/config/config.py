class Config(object):
    homeDir = "/home/quypn/IProject/face_recognition/"
    alignedImagePath = homeDir + "face/data/aligned-images"
    trainingImagePath = homeDir + "face/data/training-images"
    batchRepresent = homeDir + "batch-represent/main.lua"
    repsImagePath = homeDir + "face/data/generated-embeddings"
    alignedCache = homeDir + "face/data/aligned-images/cache.t7"

    videoPath = homeDir + "video/therock.mp4"

    # accuracy
    threshold = 0.4
