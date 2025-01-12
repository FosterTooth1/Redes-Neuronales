import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Input, Add, SeparableConv2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import classification_report
import time
from keras.regularizers import l2
from keras.applications import VGG16
from keras.applications import ResNet50
from sklearn.metrics import precision_score, recall_score, f1_score

# Cargar CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizamos los valores de los píxeles en el rango [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convertimos las etiquetas a one-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Función para construir MLP
def build_mlp(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def build_cnn(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

# Función para construir CNN con regularización
def build_cnn_regularized(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model

# Función para construir CNN con técnicas avanzadas
def build_cnn_advanced(input_shape):
    input_layer = Input(shape=input_shape)

    # Primera capa convolucional con Batch Normalization
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Segunda capa convolucional separable
    x = SeparableConv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Residual block
    residual = Conv2D(64, (1, 1), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Add()([x, residual])  # Conexión residual

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Función para construir CNN con Transfer Learning
def build_cnn_transfer_learning(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output_layer)
    return model

# Función para construir CNN con Fine Tuning
def build_cnn_fine_tuning(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:-10]:  # Desbloqueamos las últimas 10 capas
        layer.trainable = True

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output_layer)
    return model

# Función genérica para entrenar y evaluar
def train_and_evaluate(model, x_train, y_train, x_test, y_test, epochs=30, batch_size=64):
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    end_time = time.time()

    # Evaluación
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    report = classification_report(y_true, y_pred, target_names=[f'Clase {i}' for i in range(10)])
    print(report)
    print(f"Tiempo de entrenamiento: {end_time - start_time:.2f} segundos")

    return history, end_time - start_time

# Entrenar y evaluar MLP
model_mlp = build_mlp((32, 32, 3))
history_mlp, time_mlp = train_and_evaluate(model_mlp, x_train, y_train_cat, x_test, y_test_cat)

# Entrenar y evaluar CNN básica
model_cnn = build_cnn((32, 32, 3))
history_cnn, time_cnn = train_and_evaluate(model_cnn, x_train, y_train_cat, x_test, y_test_cat)

# Entrenar y evaluar CNN regularizada
model_cnn_reg = build_cnn_regularized((32, 32, 3))
history_cnn_reg, time_cnn_reg = train_and_evaluate(model_cnn_reg, x_train, y_train_cat, x_test, y_test_cat)

# Entrenar y evaluar CNN avanzada
model_cnn_adv = build_cnn_advanced((32, 32, 3))
history_cnn_adv, time_cnn_adv = train_and_evaluate(model_cnn_adv, x_train, y_train_cat, x_test, y_test_cat)

# Entrenar y evaluar CNN con Transfer Learning
model_cnn_tl = build_cnn_transfer_learning((32, 32, 3))
history_cnn_tl, time_cnn_tl = train_and_evaluate(model_cnn_tl, x_train, y_train_cat, x_test, y_test_cat)

# Entrenar y evaluar CNN con Fine Tuning
model_cnn_ft = build_cnn_fine_tuning((32, 32, 3))
history_cnn_ft, time_cnn_ft = train_and_evaluate(model_cnn_ft, x_train, y_train_cat, x_test, y_test_cat)

# Graficar precisión de entrenamiento y validación
def plot_history(histories, labels):
    plt.figure(figsize=(12, 6))
    for history, label in zip(histories, labels):
        plt.plot(history.history['accuracy'], label=f'{label} - Entrenamiento', linestyle='--')
        plt.plot(history.history['val_accuracy'], label=f'{label} - Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_history([history_mlp, history_cnn, history_cnn_reg, history_cnn_adv], ['MLP', 'CNN', 'CNN Regularizada', 'CNN Avanzada', 'CNN Transfer Learning', 'CNN Fine Tuning'])

# Resumen de resultados
results = {
    "MLP": {"time": time_mlp},
    "CNN": {"time": time_cnn},
    "CNN_Regularized": {"time": time_cnn_reg},
    "CNN_Advanced": {"time": time_cnn_adv},
    "CNN_Transfer_Learning": {"time": time_cnn_tl},
    "CNN_Fine_Tuning": {"time": time_cnn_ft}
}

print("Resultados:")
for model_name, metrics in results.items():
    print(f"{model_name}: Tiempo de entrenamiento: {metrics['time']:.2f} segundos")
    
# Implementacion de ventana deslizante
    
import cv2  # Necesario para redimensionar ventanas
# Mapeo de etiquetas para referencia
label_names = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo',
               'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']


# Clase objetivo: 'Gato'
target_class = 3
target_class_name = label_names[target_class]
print(f"Clase objetivo seleccionada: {target_class_name} (Clase {target_class})")

def sliding_window_detector(model, image, window_size=(16, 16), step_size=8, threshold=0.5):
    """
    Detecta la presencia de la clase objetivo en una imagen utilizando el algoritmo de ventana deslizante.

    Args:
        model: Modelo de clasificación entrenado.
        image: Imagen de entrada (32x32x3).
        window_size: Tamaño de la ventana deslizante.
        step_size: Paso de deslizamiento de la ventana.
        threshold: Umbral de confianza para la detección.

    Returns:
        detected: Booleano que indica si la clase fue detectada.
        detection_windows: Lista de ventanas detectadas con confianza >= threshold.
    """
    detected = False
    detection_windows = []
    img_height, img_width, _ = image.shape
    win_height, win_width = window_size

    for y in range(0, img_height - win_height + 1, step_size):
        for x in range(0, img_width - win_width + 1, step_size):
            window = image[y:y+win_height, x:x+win_width]
            # Redimensionar la ventana al tamaño de entrada del modelo (32x32)
            window_resized = cv2.resize(window, (32, 32))
            window_expanded = np.expand_dims(window_resized, axis=0)
            # Predecir la clase
            preds = model.predict(window_expanded)
            pred_class = np.argmax(preds, axis=1)[0]
            pred_prob = preds[0][target_class]
            if pred_class == target_class and pred_prob >= threshold:
                detected = True
                detection_windows.append((x, y, pred_prob))
    return detected, detection_windows

def evaluate_sliding_window_detector(model, x_test, y_test, window_size=(16, 16), step_size=8, threshold=0.5):
    """
    Evalúa el detector de ventana deslizante en el conjunto de prueba.

    Args:
        model: Modelo de clasificación entrenado.
        x_test: Conjunto de imágenes de prueba.
        y_test: Etiquetas verdaderas de prueba.
        window_size: Tamaño de la ventana deslizante.
        step_size: Paso de deslizamiento de la ventana.
        threshold: Umbral de confianza para la detección.

    Returns:
        metrics: Diccionario con precisión, recall y F1-score.
    """
    y_true = (y_test.flatten() == target_class).astype(int)
    y_pred = []

    for idx, image in enumerate(x_test):
        detected, _ = sliding_window_detector(model, image, window_size, step_size, threshold)
        y_pred.append(int(detected))
        if (idx+1) % 1000 == 0:
            print(f"Procesadas {idx+1} imágenes")

    y_pred = np.array(y_pred)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    report = classification_report(y_true, y_pred, target_names=['No Gato', 'Gato'])
    print("Reporte de Clasificación del Detector de Ventana Deslizante:")
    print(report)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return metrics

# Evaluar el detector
print("Evaluando el detector de ventana deslizante...")
metrics_detector = evaluate_sliding_window_detector(model_cnn, x_test, y_test, window_size=(16, 16), step_size=8, threshold=0.5)

print("Métricas del Detector:")
for metric, value in metrics_detector.items():
    print(f"{metric.capitalize()}: {value:.4f}")
    
def visualize_detections(model, image, detection_windows, threshold=0.5):
    '''
    Visualiza las ventanas detectadas en una imagen.

    Args:
        model: Modelo de clasificación entrenado.
        image: Imagen de entrada (32x32x3).
        detection_windows: Lista de ventanas detectadas.
        threshold: Umbral de confianza para la visualización.
    '''
    image_copy = (image * 255).astype(np.uint8).copy()
    for (x, y, prob) in detection_windows:
        cv2.rectangle(image_copy, (x, y), (x + 16, y + 16), (0, 255, 0), 1)
        cv2.putText(image_copy, f"{prob:.2f}", (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
    
    plt.figure(figsize=(2,2))
    plt.imshow(image_copy)
    plt.title(f"Detecciones de '{target_class_name}'")
    plt.axis('off')
    plt.show()

# Mostrar algunas detecciones
num_examples = 5
indices = np.where((y_test.flatten() == target_class) | (y_test.flatten() != target_class))[0][:num_examples]

for idx in indices:
    image = x_test[idx]
    true_label = y_test[idx][0]
    detected, detection_windows = sliding_window_detector(model_cnn, image, window_size=(16,16), step_size=8, threshold=0.5)
    plt.figure()
    visualize_detections(model_cnn, image, detection_windows, threshold=0.5)
    plt.title(f"Imagen {idx} - Verdadero: {'Gato' if true_label == target_class else label_names[true_label]} - Detectado: {'Sí' if detected else 'No'}")
