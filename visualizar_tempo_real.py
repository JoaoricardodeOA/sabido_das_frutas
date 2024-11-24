import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Abrir meu lindo modelo
model = load_model('modelo.h5')  
image_size = (150, 150) 

# Se houver erro na câmera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao acessar a camera.")
    exit()
#Looping de verificação com base no modelo
while True:
    ret, frame = cap.read()

    img = cv2.resize(frame, image_size)
    img = np.expand_dims(img, axis=0) 
    img = img / 255.0  

    
    predicoes = model.predict(img)
    confianca = np.max(predicoes)
    classe_predita = np.argmax(predicoes)

    
    confidence_label = f"Confiança: {confianca:.2f}"

 
    cv2.putText(frame, "Espada" if classe_predita == 0 else "Rosa", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, confidence_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Classificacao em tempo real', frame)


