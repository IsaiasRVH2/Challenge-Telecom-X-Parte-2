# 📊 Telecom X – Parte 2: Predicción de Cancelación (Churn)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)


## 🎯 Misión

Desarrollar **modelos predictivos** capaces de prever qué clientes tienen mayor probabilidad de cancelar sus servicios.

La empresa quiere anticiparse al problema de la cancelación, y es necesaro construir un pipeline robusto para esta etapa inicial de modelado.

## 🧠 Objetivos del Desafío

- ✅ Preparar los datos para el modelado (tratamiento, codificación, normalización)
- ✅ Realizar análisis de correlación y selección de variables
- ✅ Entrenar dos o más modelos de clasificación
- ✅ Evaluar el rendimiento de los modelos con métricas
- ✅ Interpretar los resultados, incluyendo la importancia de las variables
- ✅ Crear una conclusión estratégica señalando los principales factores que influyen en la cancelación

## 📁 Estructura del Proyecto

```
Challenge Telecom X - Parte 2/
│
├── datos/
│   └── datos_tratados.csv          # Dataset preprocesado
│
├── modelos/
│   ├── champion.pkl                # Modelo final entrenado
│   └── onehotencoder.pkl           # Codificador para variables categóricas
│
├── telecomx_churn_prediction.ipynb # Notebook principal con análisis completo
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Este archivo
```

## 🔬 Metodología

### 1. Preparación de Datos

- **Carga y exploración** del dataset tratado
- **Eliminación** de columnas no predictivas (`CustomerID`)
- **Codificación** de variables categóricas usando OneHotEncoder
- **Encoding** de la variable objetivo con LabelEncoder
- **Balanceo** de clases mediante técnicas de sampling

### 2. Análisis del Desbalanceo

Se identificó que solo el **26.54%** de los clientes abandonaron el servicio, lo que representa un dataset desbalanceado. Se evaluaron dos estrategias:

- **NearMiss (Undersampling)**: Recall = 0.7560 ✅
- **SMOTE (Oversampling)**: Recall = 0.7512

**Decisión**: Se seleccionó **NearMiss** por su mejor desempeño en recall.

### 3. Análisis Exploratorio

Se realizaron análisis de correlación y visualizaciones para identificar patrones:

- **Heatmap de correlaciones** entre variables numéricas
- **Boxplots** de Churn vs Tenure, ChargesTotal y ChargesMonthly
- Identificación de tendencias clave

### 4. Modelado y Evaluación

Se entrenaron y evaluaron **7 modelos** diferentes:

| Modelo | Recall | F1-Score | Accuracy | Precision |
|--------|--------|----------|----------|-----------|
| **Logistic Regression** | **0.756** | **0.713** | **0.675** | **0.675** |
| SVC | 0.751 | 0.710 | 0.673 | 0.673 |
| MLP | 0.727 | 0.694 | 0.664 | 0.664 |
| XGBoost | 0.665 | 0.674 | 0.684 | 0.684 |
| Random Forest | 0.630 | 0.656 | 0.683 | 0.683 |
| Decision Tree | 0.606 | 0.642 | 0.679 | 0.680 |
| Baseline | 0.500 | 0.333 | 0.500 | 0.250 |

**Modelo Campeón**: **Regresión Logística** con penalización L1

### 5. Selección de Características

Utilizando **regularización L1**, se redujo el modelo a las **13 variables más importantes**, manteniendo el rendimiento:

- **Recall**: 74.8%
- **F1-Score**: 71.3%
- **Accuracy**: 67.5%

## 🔑 Factores Clave Identificados

### Factores que AUMENTAN el riesgo de churn:

1. **Cargos Totales Altos** (Coef: +3.59) - Factor más crítico
2. **Contratos Mes a Mes** (Coef: +0.71) - Alta volatilidad
3. **Método de Pago: Cheque Electrónico** (Coef: +0.43) - Fricción en pagos
4. **Facturación Electrónica/Paperless** (Coef: +0.36) - Paradoja digital

### Factores que RETIENEN al cliente:

1. **Antigüedad/Tenure** (Coef: -3.96) - Protección más fuerte
2. **Servicio Telefónico** (Coef: -0.70) - Ancla de lealtad
3. **Seguridad en Línea** (Coef: -0.46) - Valor agregado
4. **Soporte Técnico** (Coef: -0.41) - Servicio diferenciador

## 💡 Estrategias de Retención Propuestas

### 1. 📝 Programa de Migración de Contratos
Diseñar campañas de **upselling** dirigidas a usuarios con contrato "Mes a Mes". Ofrecer:
- Descuento mensual por contrato anual
- Servicios extra gratuitos

### 2. 📦 Empaquetamiento Estratégico
Regalar **Soporte Técnico** y **Seguridad en Línea** durante los primeros 3-6 meses a nuevos usuarios para incrementar la percepción de valor durante la etapa crítica de retención.

### 3. 💳 Auditoría de Experiencia de Pago
Investigar el flujo de pago con "Cheque Electrónico" y promover la transición hacia métodos automáticos mediante incentivos.

### 4. 🎯 Intervención Temprana
Integrar el modelo al CRM para identificar clientes en riesgo y realizar seguimiento proactivo antes de que decidan cancelar.

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **pandas** - Manipulación de datos
- **numpy** - Operaciones numéricas
- **scikit-learn** - Modelado y preprocesamiento
- **imbalanced-learn** - Técnicas de balanceo
- **XGBoost** - Gradient boosting
- **matplotlib / seaborn** - Visualización
- **pickle** - Serialización de modelos

## 🚀 Cómo Usar Este Proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/IsaiasRVH2/Challenge-Telecom-X-Parte-2.git
cd Challenge-Telecom-X-Parte-2
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar el notebook

```bash
jupyter notebook telecomx_churn_prediction.ipynb
```

### 4. Cargar el modelo entrenado

```python
import pickle

# Cargar el codificador
with open('modelos/onehotencoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Cargar el modelo
with open('modelos/champion.pkl', 'rb') as file:
    modelo = pickle.load(file)
```

## 📊 Resultados Principales

- ✅ **Identificación exitosa del 74.8%** de clientes en riesgo de churn
- ✅ **Modelo interpretable** que permite entender por qué se van los clientes
- ✅ **Reducción a 13 variables clave** sin pérdida de rendimiento
- ✅ **Insights accionables** para el equipo de negocio

## 🎓 Aprendizajes Clave

1. **El desbalanceo importa**: Técnicas como NearMiss pueden superar significativamente al SMOTE en ciertos contextos
2. **La interpretabilidad es valiosa**: Modelos más simples (Regresión Logística) pueden ser superiores a modelos complejos cuando el negocio necesita entender el "por qué"
3. **La regularización L1 es poderosa**: Permite reducir dimensionalidad manteniendo rendimiento
4. **El recall es crítico en churn**: Es más costoso perder un cliente que no identificamos, que investigar un falso positivo

## 👤 Autor

**Isaias RVH2**
- GitHub: [@IsaiasRVH2](https://github.com/IsaiasRVH2)

## 📜 Licencia

Este proyecto es parte del programa **Oracle Next Education** y fue desarrollado con fines educativos.

---

⭐ **¡Si este proyecto te resultó útil, considera darle una estrella!** ⭐