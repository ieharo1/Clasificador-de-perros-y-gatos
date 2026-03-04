# 🐕 Clasificador de Perros y Gatos con Deep Learning

Modelo de Machine Learning para clasificación automática de imágenes utilizando redes neuronales convolucionales (CNN). Desarrollado por **Isaac Esteban Haro Torres**.

---

## 📝 Descripción

Proyecto de Deep Learning que implementa una red neuronal convolucional (CNN) para clasificar imágenes entre perros y gatos con alta precisión.

### ¿Qué hace este proyecto?

- **Entrenamiento de modelo**: Entrena una CNN desde cero con el dataset de Dogs vs Cats
- **Data Augmentation**: Aumenta el dataset con transformaciones para mejorar generalización
- **Predicción en tiempo real**: Clasifica nuevas imágenes con probabilidad de confianza
- **Visualización de resultados**: Muestra métricas de rendimiento y matrices de confusión

---

## ✨ Características Principales

| Característica | Descripción |
|----------------|-------------|
| 🧠 **CNN personalizada** | Arquitectura propia diseñada para clasificación de imágenes |
| 📈 **Data Augmentation** | Rotación, zoom, flip horizontal para expandir dataset |
| 🎯 **Transfer Learning** | Opción para usar modelos pre-entrenados (VGG16, ResNet) |
| 📊 **Métricas detalladas** | Accuracy, Precision, Recall, F1-Score |
| 🔄 **Early Stopping** | Previene overfitting automáticamente |
| 💾 **Modelos guardados** | Exporta el modelo entrenado para producción |

---

## 🛠️ Stack Tecnológico

- **Lenguaje**: Python 3.10+
- **Framework ML**: TensorFlow 2.x / Keras
- **Procesamiento de imágenes**: OpenCV, Pillow
- **Análisis de datos**: NumPy, Pandas
- **Visualización**: Matplotlib, Seaborn
- **Entorno**: Jupyter Notebook / Google Colab

---

## 🚀 Instalación y Uso

### Instalación de dependencias

```bash
pip install tensorflow opencv-python pillow numpy pandas matplotlib seaborn
```

### Entrenamiento del modelo

```python
# Cargar y preparar datos
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Entrenar
model.fit(train_generator, epochs=50, validation_data=val_generator)
```

### Predicción con nuevo imagen

```python
from tensorflow.keras.models import load_model
import cv2

model = load_model('modelo_perros_gatos.h5')

def classify_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = img.reshape(1, 150, 150, 3)
    
    prediction = model.predict(img)
    return "Perro" if prediction[0] > 0.5 else "Gato"
```

---

## 📁 Estructura del Proyecto

```
Clasificador-de-perros-y-gatos/
├── Ringa_Tech_Clasificador_de_perros_y_gatos.ipynb
├── modelo_perros_gatos.h5          # Modelo entrenado
├── requirements.txt
└── README.md
```

---

## 🏗️ Arquitectura del Modelo

```
Modelo CNN:
├── Conv2D(32, 3x3) + ReLU + MaxPool
├── Conv2D(64, 3x3) + ReLU + MaxPool
├── Conv2D(128, 3x3) + ReLU + MaxPool
├── Flatten
├── Dense(512) + ReLU + Dropout(0.5)
└── Dense(1) + Sigmoid (Salida: Perro/Gato)
```

---

## 📊 Resultados Esperados

- **Accuracy en entrenamiento**: ~95%
- **Accuracy en validación**: ~85-90%
- **Loss**: < 0.3

---

## 💡 Casos de Uso

1. **Apps de mascotas**: Integración en aplicaciones de identificación de mascotas
2. **Refugios de animales**: Automatización de clasificación de mascotas
3. **Sistemas de seguridad**: Detección de animales en propiedades
4. **Estudios biológicos**: Clasificación automática en investigaciones

---

## 🔧 Personalización

### Cambiar tamaño de imágenes

```python
IMG_SIZE = 224  # Cambiar según necesidad
```

### Usar Transfer Learning

```python
from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(weights='imagenet', include_top=False)
# Agregar capas personalizadas arriba
```

---

## 🤝 Contribuciones

¿Mejoraste el modelo? ¿Agregaste nuevas características?
¡Abre un Pull Request!

---

## 👨‍💻 Desarrollado por Isaac Esteban Haro Torres

**Ingeniero en Sistemas · Full Stack · Automatización · Data**

- 📧 Email: zackharo1@gmail.com
- 📱 WhatsApp: 098805517
- 💻 GitHub: https://github.com/ieharo1
- 🌐 Portafolio: https://ieharo1.github.io/portafolio-isaac.haro/

---

© 2026 Isaac Esteban Haro Torres - Todos los derechos reservados.
