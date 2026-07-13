# Ikusizko ereduen 3D gaitasunak ebaluatzen

El objetivo principal de esta investigación es analizar cómo influyen los diferentes paradigmas de preentrenamiento —Aprendizaje Auto-supervisado (SSL) y Aprendizaje Supervisado por Lenguaje (LSL)— en la calidad de la reconstrucción 3D densa y en la preservación de características geométricas y espaciales.



## Contribuciones
1. **CLEVR-Rec Dataset.** Un nuevo conjunto de datos sintéticos derivado de CLEVR que reformula la reconstrucción 3D como una tarea de predicción binaria de ocupación a nivel de voxel, proporcionando matrices volumétricas de resolución $64 \times 64 \times 64$. Cuenta con $85,000$ escenas para train, validación y test. El dataset se encuentra disponible en: [txenzemin/CLEVR-Rec-3D en Hugging Face](https://huggingface.co/datasets/txenzemin/CLEVR-Rec-3D).
2. **Decodificador 3D basado en DPT.** Diseño e implementación de un decodificador convolucional inspirado en la arquitectura Dense Prediction Transformer (DPT), adaptado para proyectar representaciones latentes 2D hacia un volumen discreto 3D.



## Metodología
El flujo de trabajo consta de dos fases principales: (1) la extracción de características mediante backbones congelados, (2) mapeo de las representaciones 2D al espacio 3D mediante el decodificador. En la imagen posterior se ilustra el flujo del proceso:

<img width="1404" height="768" alt="Decoder ENG" src="https://github.com/user-attachments/assets/824d4f68-abf8-4941-b93a-662c4dd9cfc3" />

Para transformar las representaciones 2D en un volumen 3D, el decodificador implementa cuatro etapas:

1. **Reassemble.** Extrae las representaciones de cuatro capas intermedias del ViT junto con el token `CLS`. Se concatena el token `CLS` a cada uno de los patch embeddings y se reestructuran los vectores resultantes en mapas de características 2D que preservan la relación espacial de la imagen original.
2. **Fusion.** Combina las representaciones de las cuatro capas del ViT mediante bloques residuales de convolución. Este proceso integra la información abstracta y los detalles finos de las escenas.
3. **2D-to-3D transformation (Lifting).** Transforma el mapa 2D en un volumen discreto inicial en 3D de baja resolución ($4 \times 4 \times 4$). Para ello, se expanden los canales hacia un espacio lineal equivalente a la resolución del voxel base, reordenando los datos en tres ejes espaciales.
4. **3D Upsampling.** Escala progresivamente el volumen inicial mediante bloques convolucionales transpuestos 3D (ConvTranspose3D). Se expande la resolución del volumen de manera jerárquica ($4^3 \rightarrow 8^3 \rightarrow 16^3 \rightarrow 32^3$) hasta alcanzar la resolución objetivo de $64 \times 64 \times 64$ vóxeles, donde cada elemento estima la probabilidad binaria de ocupación.



## Conclusiones
1. **Rendimiento cuantitativo similar.** Tanto los modelos LSL como los SSL logran métricas de Iou (Intersection over Union) globales muy cercanas (80% ~ 87%).
2. **Divergencia en escenas complejas.** Al aumentar el número de objetos y las oclusiones, los modelos SSL demuestran poseer una comprensión geométrica y una percepción espacial del entorno más robusta.

---

## Estructura del Proyecto
El repositorio está organizado de la siguiente manera:

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
