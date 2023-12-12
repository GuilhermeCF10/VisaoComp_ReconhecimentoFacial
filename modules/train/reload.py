# Função para fazer reload do model
def reloadModel():
    # Imports
    import os
    import cv2
    import numpy as np
    from .train import trainData

    print("\n\tRecarregando modelo...\n")
    trainData("./database/")
    model = cv2.dnn.readNetFromTorch(os.path.join(os.path.dirname(__file__), './model/openface.nn4.small2.v1.t7'))
    databaseEmbeddings = np.load(os.path.join(os.path.dirname(__file__), './database/databaseEmbeddings.npz'))
    return model, databaseEmbeddings