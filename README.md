# NeuroSleep

## An Explainable AI Framework for Single-Channel EEG-Based Sleep Event Detection

NeuroSleep is an experimental research project that investigates whether
a **single-channel EEG signal** can be used to identify sleep-related
respiratory events. It uses deep learning to classify short EEG segments
as either **respiratory event** or **no respiratory event**, while an
auxiliary task predicts the sleep stage.

The project also aims to incorporate Explainable AI (XAI) techniques to
highlight EEG regions or patterns that influence the model's
predictions.

> **Research limitation:** NeuroSleep is a research prototype, not a
> clinically validated diagnostic system. Its predictions must not be
> used as a medical diagnosis.

------------------------------------------------------------------------

## 1. Project objectives

1.  Prepare single-channel EEG recordings from overnight polysomnography
    (PSG) data.
2.  Divide EEG recordings into fixed-length 30-second windows.
3.  Generate respiratory-event labels from expert event annotations.
4.  Train a 1D Convolutional Neural Network (CNN) to distinguish
    respiratory-event windows from non-event windows using EEG alone.
5.  Investigate multi-task learning by adding sleep-stage classification
    as an auxiliary task.
6.  Apply XAI methods to investigate which EEG samples, regions, or
    frequency patterns influence model predictions.
7.  Develop an inference workflow that accepts an EEG EDF recording and
    produces window-level predictions without requiring respiratory or
    oxygen-saturation channels.

## 2. Dataset

### Sleep Heart Health Study (SHHS)

The project uses the **Sleep Heart Health Study (SHHS)** dataset,
obtained through the **National Sleep Research Resource (NSRR)**.

  -----------------------------------------------------------------------
  Resource                            Purpose
  ----------------------------------- -----------------------------------
  PSG EDF recordings                  Source of the EEG signal

  Respiratory-event XML annotations   Reference labels for hypopnea and
                                      obstructive apnea

  Sleep-stage XML annotations         Reference labels for Wake, N1, N2,
                                      N3 and REM

  SHHS participant dataset CSV        Participant-level metadata,
                                      including `ahi_a0h3a`

  SHHS data dictionary                Reference for interpreting dataset
                                      variables
  -----------------------------------------------------------------------

The `ahi_a0h3a` variable is available for dataset-level analysis. It is
**not an input feature** to the EEG model.

### Current processed dataset

  Item                            Value
  ------------------------ ------------
  Participants processed            145
  EEG windows generated         143,687
  Window duration            30 seconds
  Sampling rate                  125 Hz
  Samples per window              3,750
  Respiratory classes                 2
  Sleep-stage classes                 5

These counts describe the prepared dataset before the seven windows with
unknown sleep stage were excluded from the merged multi-task metadata.
The resulting multi-task metadata contains **143,680 windows**.

### Respiratory-event labels

The current project scope includes **hypopnea** and **obstructive
apnea**.

For the paper-style labeling experiment, a 30-second EEG window is
labeled positive when an individual annotated hypopnea or
obstructive-apnea event overlaps at least **10 continuous seconds** of
that window.

    Label Meaning
  ------- -------------------------------------
        0 No target respiratory event
        1 Hypopnea or obstructive-apnea event

SpO₂ desaturation and artifact annotations are not separate target
classes. Central apnea and mixed apnea are not included in the current
agreed project scope.

### Sleep-stage labels

    Model ID Sleep stage
  ---------- -------------
           0 N1
           1 N2
           2 N3
           3 REM
           4 Wake

SHHS Stage 3 and Stage 4 annotations are combined into N3.

## 3. EEG preprocessing

The EEG preprocessing workflow is designed to:

1.  Read the PSG EDF recording.
2.  Select the configured EEG channel (the processed project data uses
    the EEG channel associated with the C4/A1 derivation).
3.  Apply a 0.3--35 Hz band-pass filter.
4.  Resample the signal to 125 Hz where required by the preprocessing
    pipeline.
5.  Divide the signal into non-overlapping 30-second windows.
6.  Save the EEG windows and corresponding metadata.

Each model input contains 3,750 samples, one EEG channel, and a
30-second duration. The training pipeline applies per-window z-score
normalization.

## 4. System architecture

The intended NeuroSleep workflow is:

``` text
                  SHHS PSG EDF recordings
                            |
                            v
                  Select single EEG channel
                            |
                            v
                    EEG preprocessing
                Filter / resample / normalize
                            |
                            v
                  Divide into 30-sec windows
                            |
                            v
                 Multi-task 1D CNN model
                            |
                 Shared EEG feature extractor
                       /          \
                      v            v
             Respiratory-event   Sleep-stage
                  output           output
                      |               |
                      v               v
                Event / No event   N1 / N2 / N3 /
                                     REM / Wake
                       \             /
                        \           /
                         v         v
                       Explainable AI
                            |
                            v
                  Window-level predictions
                    and explanations
```

### Multi-task 1D CNN

The current multi-task model has a shared CNN feature extractor and two
prediction heads.

**Shared feature extractor**

-   Conv1D: 16 filters, kernel size 15, stride 2
-   Batch normalization and ELU activation
-   Max pooling
-   Conv1D: 32 filters, kernel size 11, stride 2
-   Batch normalization and ELU activation
-   Max pooling
-   Conv1D: 64 filters, kernel size 7
-   Batch normalization and ELU activation
-   Global average pooling

**Respiratory-event head**

-   Dense layer with 64 units and ReLU activation
-   Dropout
-   Sigmoid output
-   Binary classification: event / no event

**Sleep-stage head**

-   Dense layer with 64 units and ReLU activation
-   Dropout
-   Softmax output with five classes

The combined training objective uses binary cross-entropy for
respiratory-event detection and sparse categorical cross-entropy for
sleep-stage classification. The task loss weights are 0.7 for
respiratory-event detection and 0.3 for sleep-stage classification.

The sleep-stage task is an auxiliary task intended to encourage the
shared network to learn useful sleep-related EEG representations.
Whether this improves respiratory-event detection must be determined
through evaluation.

## 5. Class imbalance

The respiratory labels are imbalanced, with more no-event windows than
event windows. This can encourage a classifier to predict the majority
class too often.

A class-weighted training experiment is being conducted. The weights
calculated from the training split are approximately:

  Respiratory class     Weight
  ------------------- --------
  No event (0)          0.6462
  Event (1)             2.2068

The weighted experiment uses weighted binary cross-entropy for the
respiratory output. The sleep-stage loss remains unweighted. The purpose
is to investigate whether increasing the training penalty for event
examples improves event detection.

## 6. Participant-level data split

The project uses a participant-level split to reduce data leakage
between training, validation, and test sets.

  Split          Participants
  ------------ --------------
  Training                 87
  Validation               29
  Test                     29

All windows belonging to a participant remain within that participant's
assigned split. The test set is held out for final evaluation.

## 7. Evaluation

Respiratory-event detection is evaluated using accuracy, balanced
accuracy, ROC-AUC, precision, recall, F1-score, and a confusion matrix.

Sleep-stage classification is evaluated separately using accuracy and
balanced accuracy.

ROC-AUC measures ranking/discrimination across thresholds. Precision,
recall, F1-score, and confusion-matrix values depend on the selected
classification threshold. The threshold should be selected using
validation data, not by optimizing against the held-out test set.

### Previous experimental results

The following are results from earlier completed experiments. They are
not claims about the current weighted multi-task experiment.

  -----------------------------------------------------------------------
  Experiment          Test accuracy  Balanced accuracy       Test ROC-AUC
  -------------- ------------------ ------------------ ------------------
  Initial                    0.6126             0.6165             0.6576
  respiratory                                          
  CNN                                                  

  Refined                    0.6358             0.6196             0.6630
  respiratory                                          
  CNN (30%                                             
  overlap rule)                                        

  Paper-style                0.6729             0.6057             0.6562
  respiratory                                          
  CNN                                                  

  Initial                    0.7353             0.5040             0.6536
  unweighted                                           
  multi-task CNN                                       
  -----------------------------------------------------------------------

The initial unweighted multi-task model achieved 70.50% sleep-stage test
accuracy. Its respiratory recall at a threshold of 0.5 was very low,
showing why accuracy alone is insufficient for this imbalanced task.

The class-weighted multi-task experiment is in progress. Its final
performance has not yet been established.

## 8. Explainable AI (planned)

Explainability is a core objective of NeuroSleep. The planned work is to
investigate methods that can help show which parts of the EEG signal
influenced a prediction.

Potential methods include:

-   **Integrated Gradients:** Attribute model output to input EEG
    samples.
-   **SHAP:** Estimate feature contributions to model predictions.
-   **Frequency-band analysis:** Investigate the contribution of EEG
    activity in different frequency ranges.
-   **Signal-region visualization:** Display important portions of a
    30-second EEG window alongside the prediction.

These methods are planned components; they should only be described as
implemented after the corresponding code and experiments have been
completed. Explanations indicate model behavior and do not establish
physiological causation.

## 9. Inference objective

The intended inference workflow is:

1.  User supplies a PSG EDF recording containing the supported EEG
    channel.
2.  The system selects and preprocesses the EEG signal.
3.  The signal is divided into 30-second windows.
4.  Each window is passed to the trained model.
5.  The respiratory head returns an event probability and a class
    prediction.
6.  The sleep-stage head returns a sleep-stage prediction.
7.  The XAI component, once implemented, provides an explanation for the
    prediction.

**Inference is intended to use EEG alone.** Respiratory-event XML
annotations and sleep-stage annotations are used to create reference
labels for training and evaluation. They are not required as model
inputs when processing a new recording.

## 10. Project structure

The following is the intended/known organization of the project. Some
files may vary as experiments evolve.

``` text
neurosleep/
|
|-- data/
|   |-- raw/
|   |   |-- shhs/
|   |       |-- polysomnography/
|   |           |-- edfs/
|   |           |   |-- shhs1/
|   |           |
|   |           |-- annotations-events-nsrr/
|   |               |-- shhs1/
|   |
|   |-- processed/
|       |-- edf_eeg_windows/
|       |   |-- metadata.csv
|       |   |-- *.npy
|       |
|       |-- respiratory_event_dataset/
|           |-- event_metadata.csv
|           |-- refined_event_metadata.csv
|           |-- paper_style_event_metadata.csv
|           |-- sleep_stage_metadata.csv
|           |-- multitask_metadata.csv
|           |-- participant_split.csv
|
|-- models/
|   |-- neurosleep_multitask_cnn.keras
|   |-- neurosleep_multitask_weighted.keras
|
|-- preprocess_eeg_edf.py
|-- create_event_dataset.py
|-- create_event_split.py
|-- create_refined_event_dataset.py
|-- create_paper_style_dataset.py
|-- create_sleep_stage_labels.py
|-- merge_multitask_metadata.py
|-- train_respiratory_cnn.py
|-- train_refined_respiratory_cnn.py
|-- train_paper_cnn.py
|-- train_multitask_cnn.py
|-- train_multitask_weighted.py
|
|-- README.md
```

This tree is a guide to the project organization, not a guarantee that
every generated file is present in every working copy.

## 11. Environment and dependencies

The project has been developed in Python on Windows. The recorded
working environment includes Python 3.12 and TensorFlow 2.21.

Main libraries:

-   TensorFlow / Keras
-   NumPy
-   Pandas
-   SciPy
-   scikit-learn
-   MNE
-   Matplotlib (for visualizations, where used)

### Setup

Create and activate a virtual environment:

``` bash
python -m venv .venv
```

Windows CMD:

``` cmd
.venv\Scripts\activate
```

Install the dependencies:

``` bash
pip install numpy pandas scipy scikit-learn tensorflow mne matplotlib
```

Exact package versions should be recorded in a `requirements.txt` file
for reproducible experiments.

## 12. Running the project

Run commands from the project root directory.

### EEG preprocessing

``` bash
python preprocess_eeg_edf.py
```

### Generate respiratory-event labels

``` bash
python create_event_dataset.py
```

### Create participant splits

``` bash
python create_event_split.py
```

### Generate paper-style respiratory labels

``` bash
python create_paper_style_dataset.py
```

### Generate sleep-stage labels and multi-task metadata

``` bash
python create_sleep_stage_labels.py
python merge_multitask_metadata.py
```

### Train the initial multi-task CNN

``` bash
python train_multitask_cnn.py
```

### Train the class-weighted multi-task CNN

``` bash
python train_multitask_weighted.py
```

The weighted experiment is configured to save to:

``` text
models/neurosleep_multitask_weighted.keras
```

Training scripts and preprocessing scripts should be run in the
appropriate order and against the matching metadata files. Do not
overwrite earlier model checkpoints when comparing experiments.

## 13. Current project status

  -----------------------------------------------------------------------
  Component                           Status
  ----------------------------------- -----------------------------------
  SHHS EDF and XML data preparation   Completed for current
                                      145-participant subset

  EEG preprocessing and window        Completed
  generation                          

  Respiratory-event label generation  Completed

  Participant-level split             Completed

  Sleep-stage label generation        Completed

  Initial respiratory CNN experiments Completed

  Initial multi-task CNN              Completed

  Class-weighted multi-task CNN       In progress

  Threshold selection and detailed    Pending for weighted model
  comparison                          

  Explainable AI implementation       Planned

  EEG-only inference interface        Planned

  Final end-to-end validation         Pending
  -----------------------------------------------------------------------

## 14. Scope and limitations

-   The current respiratory target includes hypopnea and obstructive
    apnea only.
-   The model uses a single EEG channel; other physiological channels
    are not model inputs.
-   Event labels are derived from SHHS annotations and the project's
    stated window-overlap rule.
-   Model performance is limited by dataset size, class imbalance,
    annotation definitions, signal variability, and generalization to
    unseen participants.
-   The project has not established clinical validity or suitability for
    diagnosis.
-   Sleep-stage prediction is an auxiliary research task, not a
    substitute for expert sleep staging.
-   XAI methods explain aspects of model behavior and should not be
    interpreted as proof of causal physiological mechanisms.

## 15. Expected contribution

NeuroSleep aims to provide an experimental framework for studying:

1.  Respiratory-event detection from a single EEG channel.
2.  Multi-task learning between respiratory-event detection and
    sleep-stage classification.
3.  Class-imbalance handling in EEG-based event detection.
4.  Explainability of deep-learning predictions on EEG signals.
5.  The feasibility and limitations of EEG-only inference for
    sleep-related respiratory-event screening research.

------------------------------------------------------------------------

**Project name:** NeuroSleep\
**Project type:** Academic research project\
**Primary domain:** Deep Learning, EEG Signal Processing, Sleep
Research, Explainable AI\
**Dataset:** Sleep Heart Health Study (SHHS), via NSRR\
**Primary input:** Single-channel EEG\
**Primary output:** Respiratory event / No respiratory event\
**Auxiliary output:** Five-class sleep-stage prediction
