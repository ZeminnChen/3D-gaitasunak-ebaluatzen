# Ikusizko ereduen 3D gaitasunak ebaluatzen (CLEVR-Rec Voxel Generator)


El proyecto se centra en evaluar la comprensión y el conocimiento espacial tridimensional de los Modelos Fundacionales de Visión (*Vision Foundation Models*), 
analizando el impacto de sus paradigmas de preentrenamiento (aprendizaje auto-supervisado frente a aprendizaje supervisado por lenguaje) a través de tarea de reconstrucción volumétrica.

## Contexto

Para llevar a cabo la evaluación, el problema de la reconstrucción 3D a partir de una única imagen 2D se ha reformulado como una tarea de **predicción binaria de ocupación a nivel de voxel** (donde `1` representa voxel ocupado y `0` representa fondo o espacio vacío). 
A tales efectos:
1. Se ha construido el dataset sintético **CLEVR-Rec** (derivado de CLEVR), proporcionando matrices volumétricas de resolución $64 \times 64 \times 64$.
2. Las representaciones intermedias de los modelos visuales evaluados se mapean al espacio 3D discreto utilizando un decodificador personalizado basado en la arquitectura **Dense Prediction Transformer (DPT)**.


---

## Estructura del Proyecto
El repositorio está organizado de manera modular para separar la gestión de datos, la arquitectura de red de los modelos y las canalizaciones de evaluación:

```text
3D-gaitasunak-ebaluatzen/
│
├── code/
│   └── ablation_study
|       ├── average_pooling.py
|       ├── single_layer.py
│   ├── decoder.py               
│   └── display_predictions.ipynb
│   └── voxel_generator.py
├── perception_models
├── results/
│   └── ablation_study
│   └── frozen
├── requirements.txt         
└── README.md                
