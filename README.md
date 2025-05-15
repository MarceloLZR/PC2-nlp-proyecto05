#  Proyecto 5: Pre-entrenamiento Autodidacta y Fine-Tuning Eficiente

Este proyecto explora técnicas modernas de pre-entrenamiento de lenguaje (MLM y NSP) y adaptación a tareas específicas mediante estrategias eficientes como **Adapters** y **LoRA**, usando un modelo tipo BERT.

---

##  Objetivos

- Implementar pre-entrenamiento con **Masked Language Modeling (MLM)** y **Next Sentence Prediction (NSP)**.
- Aplicar técnicas de fine-tuning completas y eficientes:
  - Full fine-tuning.
  - Adapters (Houlsby et al., 2019).
  - LoRA (Hu et al., 2021).
- Comparar desempeño, eficiencia y uso de recursos.
- Analizar el trade-off entre número de parámetros entrenados y precisión.

---

##  Dataset

- **Pre-entrenamiento**: Wikipedia simplificado (~100M tokens).
- **Fine-tuning**: SST-2 (Stanford Sentiment Treebank v2), tarea de clasificación de sentimientos.

---
