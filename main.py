# Imports
import os
import cv2
import numpy as np
from modules.detection import detectShoulders, detectFace # Detection
from modules.image import drawRectangle
from modules.recognition import recognizePerson
from modules.train import extractEmbeddings, trainData 
from modules.store import storePersonFrame
import keyboard


import time
import uuid
# myuuid = uuid.uuid4()
# print('Your UUID is: ' + str(myuuid))
needTrain = True
if (needTrain == True):
    trainData("./database/")

# Carregar modelo e banco de dados de embeddings
model = cv2.dnn.readNetFromTorch(os.path.join(os.path.dirname(__file__), './modules/train/model/openface.nn4.small2.v1.t7'))
databaseEmbeddings = np.load("./modules/train/database/databaseEmbeddings.npz")

cap = cv2.VideoCapture(0)

MODE = "RECOGNITION"  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if keyboard.is_pressed('r'):
        MODE = "RECOGNITION"
    
    elif keyboard.is_pressed('n'):
        MODE = "REGISTER NEW PERSON"
  

    if MODE == "RECOGNITION":
        shouldersRoi = detectShoulders(frame)
        faces = detectFace(frame, roi=shouldersRoi)

        # for (x, y, w, h) in faces:
        # Considera apenas uma pessoa
        if len(faces)>0:
            (x, y, w, h) = faces[0]

            # Extrair a região do rosto
            faceRegion = frame[y:y + h, x:x + w]
            
            # Extrair o embedding do rosto
            faceEmbedding = extractEmbeddings(model, faceRegion).flatten()

            # Reconhecer a pessoa a partir do embedding
            label = recognizePerson(faceEmbedding, databaseEmbeddings)
            print(label)

            # Desenhar retângulo e label no frame
            frame = drawRectangle(frame, label, x, y, w, h)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    elif MODE == "REGISTER NEW PERSON":
        time.sleep(5)
        frameCount = 1

        personName = str(input("Insira seu nome e sobrenome: ")).replace(" ", "_")
        savePath = os.path.join(os.path.dirname(__file__), f'./database/{personName}_{uuid.uuid4()}')
        print("Capturando imagens. Aguarde 5 segundos para a captura começar...")

        
        while frameCount <= 10:  # Capturar frames por 3 segundos (30 fps)
            ret, frame = cap.read()
            if not ret:
                break

            # Detecção de ombros
            shouldersRoi = detectShoulders(frame)


            # Verifique se os ombros foram detectados
            if shouldersRoi is not None:
                # Detecção facial
                faces = detectFace(frame, roi=shouldersRoi)

                if len(faces) > 0:
                    (x, y, w, h) = faces[0]  # Pega o primeiro rosto detectado
                    faceRegion = frame[y:y + h, x:x + w]  # Recorta a região do rosto

                    # Certifique-se de que a região do rosto não está vazia antes de salvar
                    if faceRegion is not None and not faceRegion.size == 0:
                        storePersonFrame(faceRegion, frameCount, savePath)  # Salva apenas a região do rosto

                else:
                    print("Não está detectando sua face")

                frameCount += 1
            else:
                print("Não está detectando seus ombros")
            
        print("Imagens capturadas com sucesso!")
        cap.release()  # Encerrar a câmera após a captura

cap.release()
cv2.destroyAllWindows()