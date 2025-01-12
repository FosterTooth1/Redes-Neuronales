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
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Cargar CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizar los valores de los píxeles en el rango [0, 1]
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

# Función para construir CNN básica
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
