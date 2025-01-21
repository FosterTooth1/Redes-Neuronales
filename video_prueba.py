import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten,  Conv2D, MaxPooling2D, BatchNormalization, Input, Add, SeparableConv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definir el número de clases para clasificación binaria
NUM_CLASES = 2

# Cargar CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Índice de la clase "barco" en CIFAR-10
indice_barco = 8  # "ship" es la clase con índice 8

# Etiquetas: 1 para "barco", 0 para "no barco"
y_train_bin = np.where(y_train.flatten() == indice_barco, 1, 0)
y_test_bin = np.where(y_test.flatten() == indice_barco, 1, 0)

# Normalizar los valores de los píxeles en el rango [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convertir las etiquetas a one-hot encoding
y_train_cat = to_categorical(y_train_bin, NUM_CLASES)
y_test_cat = to_categorical(y_test_bin, NUM_CLASES)

# Función para ajustar una imagen a 32x32
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (32, 32))
    normalized_frame = resized_frame.astype('float32') / 255.0
    return normalized_frame

# Función para implementar el detector con ventana deslizante
def sliding_window_detector(video_path, model, window_size=(758, 758), step_size=32, threshold=0.5):
    video = cv2.VideoCapture(video_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # Recorrer la imagen con ventana deslizante
        for y in range(0, frame_height - window_size[1], step_size):
            for x in range(0, frame_width - window_size[0], step_size):
                window = frame[y:y + window_size[1], x:x + window_size[0]]
                if window.shape[:2] != window_size:
                    continue
                
                processed_window = preprocess_frame(window)
                processed_window = processed_window.reshape((1, 32, 32, 3))
                
                prediction = model.predict(processed_window)
                predicted_class = prediction.argmax(axis=1)[0]
                
                if predicted_class == 1:
                    cv2.rectangle(frame, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
        
        cv2.imshow('Detections', frame)
        
        # Si presionas 'q', rompemos el bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video.release()
    
    # Mantenemos la ventana abierta hasta que se cierre con la X o con la tecla 'q'
    while True:
        # Verifica si la ventana sigue abierta
        if cv2.getWindowProperty('Detections', cv2.WND_PROP_VISIBLE) < 1:
            break
        
        # Si se presiona 'q', también salimos
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


# Función para construir CNN avanzada utilizando la API Funcional
def build_cnn(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    
    # Primera capa convolucional con Batch Normalization
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Segunda capa convolucional separable con Batch Normalization
    x = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Residual block
    residual = Conv2D(64, (1, 1), padding='same')(x)
    res = BatchNormalization()(residual)
    
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Add()([x, res])  # Conexión residual
    x = BatchNormalization()(x)
    
    # Capa de Flatten y Densa
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Construir y compilar el modelo
model_barco = build_cnn((32, 32, 3), NUM_CLASES)
optimizer = Adam(learning_rate=0.001)
model_barco.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# (Opcional) Aumento de Datos para mejorar la generalización
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

datagen.fit(x_train)

# Entrenamiento del modelo con aumento de datos
model_barco.fit(
    datagen.flow(x_train, y_train_cat, batch_size=64),
    epochs=10,
    validation_data=(x_test, y_test_cat),
    verbose=2
)

# Evaluación del modelo
y_pred = model_barco.predict(x_test)
y_pred_classes = y_pred.argmax(axis=1)  # Para 'softmax'

precision = precision_score(y_test_bin, y_pred_classes)
recall = recall_score(y_test_bin, y_pred_classes)
f1 = f1_score(y_test_bin, y_pred_classes)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

# Ejecutar el detector en el video
sliding_window_detector('barco.mp4', model_barco)
