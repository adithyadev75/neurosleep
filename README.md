&#x09;# NeuroSleep



\## Explainable AI Framework for Single-Channel EEG-Based Sleep Event Detection



\### Project Overview



NeuroSleep is a research project focused on developing an Explainable AI framework for detecting sleep events from a single-channel EEG signal.



The project uses the Sleep Heart Health Study (SHHS) dataset and investigates deep learning methods for automated sleep-event detection.



The framework is intended to provide both:



1\. Automated sleep-event predictions from a single EEG signal.

2\. Explainable outputs that helps to identify the EEG regions contributing to model predictions.



\### Research Pipeline



```text

SHHS Dataset

&#x20;    ↓

Single-Channel EEG Selection

&#x20;    ↓

EEG Preprocessing

&#x20;    ↓

Annotation Processing

&#x20;    ↓

Dataset Construction

&#x20;    ↓

Deep Learning Model

&#x20;    ↓

Sleep Event Prediction

&#x20;    ↓

Model Evaluation

&#x20;    ↓

Explainability

```



\### Project Structure



```text

neurosleep/

│

├── data/

│   ├── raw/

│   ├── interim/

│   └── processed/

│

├── notebooks/

│

├── src/

│   ├── data/

│   ├── models/

│   ├── training/

│   ├── explainability/

│   └── utils/

│

├── configs/

├── experiments/

├── results/

│   ├── figures/

│   ├── metrics/

│   └── models/

│

├── tests/

│

├── requirements.txt

├── README.md

└── .gitignore

```



\### Dataset



The project uses the Sleep Heart Health Study (SHHS) dataset.



Raw dataset files are intentionally excluded from this repository. Dataset acquisition and preparation instructions will be documented separately.



\### Development Status



\* \[x] Project structure

\* \[ ] Dataset acquisition

\* \[ ] SHHS data exploration

\* \[ ] EEG preprocessing

\* \[ ] Annotation analysis

\* \[ ] Dataset construction

\* \[ ] Baseline model

\* \[ ] Model evaluation

\* \[ ] Explainability

\* \[ ] Final experiments

\* \[ ] Research documentation



\### Reproducibility



Experiments will be tracked using Git commits, configuration files, experiment records, and saved evaluation metrics.



\### Authors



\* Adithya Dev

\* Anshad Zaman



