from face_adding.core.face.classifier2 import *

if __name__ == '__main__':
    args, align, net = init()
    infer(args, align, net)
