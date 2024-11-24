import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D,Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array,load_img


# Mesmo sendo uma classificação binária, decidir fazer como se fosse categórica

# Aumentar fotos translação, girar, contraste,brilho,zoom, rotação e escala. Dados pequenos, logo, necessário
data_augmentation = tf.keras.Sequential([
    
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomFlip("horizontal"),
    layers.RandomContrast(0.2),  
    layers.RandomBrightness(0.2),
    layers.RandomFlip("vertical"),
    layers.RandomZoom(0.2),
    layers.RandomRotation(0.3),
    layers.Rescaling(1./255),
])

# Treino e validação usando nova biblioteca do TF 80% treino / 20% validação / categorico
treino_dataset = image_dataset_from_directory(
    "DATASET",
    labels="inferred",
    label_mode="categorical",  
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(150,150),
    batch_size=16
)
val_dataset = image_dataset_from_directory(
    "DATASET",
    labels="inferred",
    label_mode="categorical",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(150,150),
    batch_size=16
)

# Feito augmentation no treino e reescalado treino e validação para performance
treino_dataset = treino_dataset.map(lambda x, y: (data_augmentation(x), y))
val_dataset = val_dataset.map(lambda x, y: (layers.Rescaling(1./255)(x), y))

# modelo
model = Sequential([
    # Parte de convolução e MaxPool
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    #Alinhar dados em vetores
    Flatten(),
    # Camada densa para inferências, adicionado dropout para evitar overfitting
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
               metrics=['accuracy']) # ADAM GENERALIZA MAIS E AJUSTA APRENDIZADO, REDUZIDO A TAXA INICIAL DE APRENDIZADO

#5 épocas já foi perfeito
model.fit(treino_dataset, epochs=5, validation_data=val_dataset)

#Manga que parece espada
img = load_img('TESTE/IMG_20210629_184248.jpg',target_size=(150,150))

#Mostra foto
plt.imshow(img)
plt.axis("on")
plt.show()

# Pega imagem, escala prediz e pega o valor com maior porcentagem e responde no console o tipo de mangA
img = img_to_array(img)
img = np.expand_dims(img, axis=0)/255.0
probabilidades = model.predict(img)
classe_predita = np.argmax(probabilidades, axis=1)
predicted_confidence = probabilidades[0][classe_predita[0]]
print(f"Confiança: {predicted_confidence:.2f}")
print("Espada" if classe_predita == 0 else "Rosa")

#Manga Rosa muito parecida com a do vídeo
img = load_img('TESTE/frame_0002.jpg',target_size=(150,150))

#Mostra foto
plt.imshow(img)
plt.axis("on")
plt.show()

# Pega imagem, escala prediz e pega o valor com maior porcentagem e responde no console o tipo de mangA
img = img_to_array(img)
img = np.expand_dims(img, axis=0)/255.0
probabilidades = model.predict(img)
classe_predita = np.argmax(probabilidades, axis=1)
predicted_confidence = probabilidades[0][classe_predita[0]]
print(f"Confiança: {predicted_confidence:.2f}")
print("Espada" if classe_predita == 0 else "Rosa")

#Salva o lindo modelo
model.save('modelo.h5')




