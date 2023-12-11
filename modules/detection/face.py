def detectFace(image, roi=None):
    # Imports
    import cv2

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if roi is not None:
        grayRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(grayRoi, 1.1, 4)
        # Ajustar coordenadas do rosto para corresponder à posição na imagem original
        faces = [(x + roi.shape[1], y + roi.shape[0], w, h) for (x, y, w, h) in faces]
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    return faces
