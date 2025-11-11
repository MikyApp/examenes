import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils #Nos ayuda a dibujar los 21 puntos y conecciones
mp_hands = mp.solutions.hands


with mp_hands.Hands(
    static_image_mode = True, #Cuando se le asigna false trata a las imagenes de entrada como un videostream
    max_num_hands = 2, #Numero maximo de 
    min_detection_confidence = 0.5) as hands:

    image = cv2.imread('manos1.webp')
    heiht, width, _ = image.shape
    image = cv2.flip(image, 1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    #HANDEDNESS
    #print("Handedness: ", results.multi_handedness)
    #HAND LANDMARKS 
    #print("Hand landmarks: ", results.multi_hand_landmarks)

    if results.multi_hand_landmarks is not None:
        #--------------------------
        #Accediendo a los puntos 
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,255,0), thickness=4, circle_radius=5), #Cambiar el color de los puntos
                mp_drawing.DrawingSpec(color=(255,0,255), thickness=4), #Cambiar el color de las conxiones


            )
            #print(hand_landmarks)

    image = cv2.flip(image, 1)

cv2.imshow('Imagen', image)
cv2.waitKey(0)
cv2.destroyAllWindows(0)