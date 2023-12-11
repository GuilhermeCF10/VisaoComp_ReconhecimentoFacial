def detectShoulders(image):
    # Imports
    import cv2

    upperBodyCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    upperBodies = upperBodyCascade.detectMultiScale(gray, 1.1, 3)

    if len(upperBodies) == 0:
        return None

    x, y, w, h = upperBodies[0]  # Pegando a primeira detecção
    shouldersRoi = image[y:y+h, x:x+w]

    return shouldersRoi
