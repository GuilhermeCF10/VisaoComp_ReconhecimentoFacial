def drawRectangle(frame, label, x, y, w, h):
    # Imports
    import cv2

    # Desconstrução de label
    person, percentage = label

    # Desenhar o retângulo ao redor da face
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Preparar o texto da pessoa
    if person:
        cv2.putText(frame, str(person)[:40], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Preparar o texto da porcentagem, se houver
    if percentage is not None and percentage > 75:
        percentText = f"{percentage:.2f}%"  # Formatar como uma string com duas casas decimais
        cv2.putText(frame, percentText, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame
