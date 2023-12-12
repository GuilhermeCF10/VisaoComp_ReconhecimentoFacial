def trainData(databasePath):
    # Imports
    import os
    import cv2
    import numpy as np
    from tqdm import tqdm
    from .face import extractEmbeddings

    # Init
    model = cv2.dnn.readNetFromTorch(os.path.join(os.path.dirname(__file__), './model/openface.nn4.small2.v1.t7'))
    databaseEmbeddings = {}

    os.system("clear")
    print("\n\tTreinando modelo...\n")

    personNames = os.listdir(databasePath)
    for personName in tqdm(personNames, desc="Processando pessoas"):
        personFolder = os.path.join(databasePath, personName)

        # Verifique se é um diretório
        if os.path.isdir(personFolder):
            embeddings = []
            # Remover o UUID e substituir underscores por espaços
            personNameFormatted = " ".join(personName.split("_")[:-1])

            for imageName in os.listdir(personFolder):
                imagePath = os.path.join(personFolder, imageName)
                image = cv2.imread(imagePath)

                # Verifique se a imagem foi carregada corretamente
                if image is not None:
                    embedding = extractEmbeddings(model, image)
                    embeddings.append(embedding.flatten())
                else:
                    print(f"Não foi possível carregar a imagem: {imagePath}")

                # Para entender quais arquivos estão sendo processados
                # print(imagePath)
            databaseEmbeddings[personNameFormatted] = np.array(embeddings)

    # Adicionado espaço no terminal
    print("\n\n")
    
    # Salvando os embeddings
    np.savez_compressed(os.path.join(os.path.dirname(__file__), './database/databaseEmbeddings.npz'), **databaseEmbeddings)