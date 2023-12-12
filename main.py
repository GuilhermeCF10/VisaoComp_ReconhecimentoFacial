# Imports
import os
import cv2
import numpy as np
import time
import uuid
from modules.detection import detectShoulders, detectFace # Detection
from modules.image import drawRectangle
from modules.recognition import recognizePerson
from modules.train import extractEmbeddings, trainData 
from modules.store import storePersonFrame

# Função principal
def main():
    trainData("./database/")

    model = cv2.dnn.readNetFromTorch(os.path.join(os.path.dirname(__file__), './modules/train/model/openface.nn4.small2.v1.t7'))
    databaseEmbeddings = np.load("./modules/train/database/databaseEmbeddings.npz")

    cap = cv2.VideoCapture(0)
    MODE = "RECOGNITION"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            cap.release()
            cv2.destroyAllWindows()
            registerNewPerson()
            cap = cv2.VideoCapture(0)  # Reiniciar a captura da câmera
            model, databaseEmbeddings = reloadModel()
            print("\n\tModelo de reconhecimento ativo.\n")
            continue

    
        elif key == ord('r'):
            model, databaseEmbeddings = reloadModel()
            print("\n\tModelo de reconhecimento ativo.\n")
            continue

        shouldersRoi = detectShoulders(frame)
        faces = detectFace(frame, roi=shouldersRoi)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Considera apenas um rosto

            # Extrair a região do rosto e o embedding
            faceRegion = frame[y:y + h, x:x + w]
            faceEmbedding = extractEmbeddings(model, faceRegion).flatten()

            # Reconhecer a pessoa a partir do embedding
            label = recognizePerson(faceEmbedding, databaseEmbeddings)

            # Desenhar retângulo e label no frame
            frame = drawRectangle(frame, label, x, y, w, h)


        cv2.imshow('Reconhecimento', frame)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Função para o modo de registro de nova pessoa
def registerNewPerson():
    personName = input("Insira seu nome e sobrenome: ").replace(" ", "_")
    savePath = os.path.join(os.path.dirname(__file__), f'./database/{personName}_{str(uuid.uuid4())}')
    os.makedirs(savePath, exist_ok=True)

    cap = cv2.VideoCapture(0)
    frameCount = 0

    print("Prepare-se para a captura de imagens.")
    while frameCount < 10:
        ret, frame = cap.read()
        if not ret:
            break

        # Detecção de ombros
        shouldersRoi = detectShoulders(frame)
        # Detecção facial
        faces = detectFace(frame, roi=shouldersRoi)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Pega o primeiro rosto detectado
            faceRegion = frame[y:y + h, x:x + w]  # Recorta a região do rosto          
            storePersonFrame(faceRegion, frameCount, savePath)  # Salva apenas a região do rosto

        else:
            print("Não está detectando sua face")


        cv2.imshow('Registro', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frameCount += 1

    cap.release()
    cv2.destroyAllWindows()

# Função para fazer reload do model
def reloadModel():
    print("\n\tRecarregando modelo...\n")
    trainData("./database/")
    model = cv2.dnn.readNetFromTorch(os.path.join(os.path.dirname(__file__), './modules/train/model/openface.nn4.small2.v1.t7'))
    databaseEmbeddings = np.load("./modules/train/database/databaseEmbeddings.npz")
    return model, databaseEmbeddings


main()