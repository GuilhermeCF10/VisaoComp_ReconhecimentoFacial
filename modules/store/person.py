def storePersonFrame(frame, frameCount, savePath):
    """
    Salva um frame como imagem JPG em um caminho específico.

    Args:
        frame: O frame a ser salvo como imagem.
        frameCount: O número do frame ou nome do arquivo.
        savePath: O caminho onde a imagem será salva.
    """
    # Imports
    import os
    import cv2

    # Certifique-se de que o diretório de destino existe, se não, crie-o
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    # Crie o nome do arquivo usando o número do frame
    file_name = f"{frameCount}.jpg"
    
    # Combine o caminho do diretório e o nome do arquivo
    file_path = os.path.join(savePath, file_name)
    
    # Salve o frame como uma imagem JPG
    cv2.imwrite(file_path, frame)

# # Certifique-se de que o frame não esteja vazio
# if frame is not None and not frame.size == 0:
#     # Crie o nome do arquivo usando o número do frame
#     file_name = f"{frameCount}.jpg"

#     # Combine o caminho do diretório e o nome do arquivo
#     file_path = os.path.join(savePath, file_name)

#     # Salve o frame como uma imagem JPG
#     cv2.imwrite(file_path, frame)