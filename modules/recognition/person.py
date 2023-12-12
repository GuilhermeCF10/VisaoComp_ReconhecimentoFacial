def recognizePerson(faceEmbedding, databaseEmbeddings, threshold=0.2):
    # Imports
    import numpy as np

    # Init
    minDistance = float('inf')
    recognizedPerson = "Desconhecido"
    confidence = 0

    # Iterar sobre cada pessoa no banco de dados
    for personName, embeddings in databaseEmbeddings.items():
        # Calcular a distância entre o embedding do rosto e cada embedding da pessoa
        distances = np.linalg.norm(embeddings - faceEmbedding, axis=1)
        minDbDistance = np.min(distances)

        # Encontrar a pessoa com a menor distância
        if minDbDistance < minDistance:
            minDistance = minDbDistance
            recognizedPerson = personName

    # Calcular a confiança baseada na distância
    if minDistance < threshold:
        confidence = (1 - minDistance / threshold) * 100
        if confidence < 75:  # Adicionar verificação de confiança extra
            recognizedPerson = "Desconhecido"
    else:
        recognizedPerson = "Desconhecido"
        confidence = 0

    return [recognizedPerson, confidence]
