def extractEmbeddings(model, face):
    """
    Função para extrair embeddings de uma imagem de rosto.
    """
    # Imports
    import cv2
    
    # Criação de um blob a partir da imagem
    faceBlob = cv2.dnn.blobFromImage(face, 1.0, (96, 96), (104.0, 177.0, 123.0), swapRB=True, crop=False)
    
    # Configurando o blob como entrada do modelo
    model.setInput(faceBlob)
    
    # Realizando a inferência para obter os embeddings
    return model.forward()