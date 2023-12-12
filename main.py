# Imports
import os
import cv2
import uuid
from modules.detection import detectShoulders, detectFace # Detection
from modules.image import drawRectangle
from modules.recognition import recognizePerson
from modules.train import extractEmbeddings, reloadModel
from modules.store import storePersonFrame

# Função principal
def main():
    model, databaseEmbeddings = reloadModel()

    cap = cv2.VideoCapture(0)
    # Melhorar processamento
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF

        # Criar novo perfil
        if key == ord('n'):
            cap.release()
            cv2.destroyAllWindows()
            registerNewPerson()
            cap = cv2.VideoCapture(0)  # Reiniciar a captura da câmera
            model, databaseEmbeddings = reloadModel()
            print("\n\tModelo de reconhecimento ativo.\n")
            continue

        # Forçar recarregamento do modelo
        elif key == ord('r'):
            model, databaseEmbeddings = reloadModel()
            print("\n\tModelo de reconhecimento ativo.\n")
            continue

        elif key == ord("t"):
            playVideo(os.path.join(os.path.dirname(__file__), './modules/tutorial/tutorial.mp4'))

        shouldersRoi = detectShoulders(frame)
        faces = detectFace(frame, roi=shouldersRoi)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Considera apenas um rosto

            # Extrair a região do rosto e o embedding
            faceRegion = frame[y:y + h, x:x + w]

            # Verifique se a região do rosto não está vazia antes de continuar
            if faceRegion.size != 0:
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

def playVideo(videoPath):
    # Criar um objeto de captura de vídeo
    cap = cv2.VideoCapture(videoPath)

    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o arquivo de vídeo.")
        return

    # Ler e exibir cada frame do vídeo
    while True:
        ret, frame = cap.read()

        # Verificar se ainda há frames no vídeo
        if not ret:
            print("Fim do vídeo.")
            break

        # Exibir o frame
        cv2.imshow("Video Player", frame)

        # Aguardar 25ms e verificar se o usuário pressionou a tecla 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Liberar o objeto de captura e fechar todas as janelas
    cap.release()
    cv2.destroyAllWindows()




# Função para o modo de registro de nova pessoa
def registerNewPerson():
    personName = input("Insira seu nome e sobrenome: ").replace(" ", "_")
    savePath = os.path.join(os.path.dirname(__file__), f'./database/{personName}_{str(uuid.uuid4())}')
    os.makedirs(savePath, exist_ok=True)

    cap = cv2.VideoCapture(0)

     # Melhorar processamento
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frameCount = 0

    print("Prepare-se para a captura de imagens.")
    while frameCount < 60:
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

            # Verifique se a região do rosto não está vazia antes de salvar
            if faceRegion is not None and faceRegion.size != 0:
                storePersonFrame(faceRegion, frameCount, savePath)
                frameCount += 1

        else:
            print("Não está detectando sua face")

        cv2.imshow('Registro', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()