# Voice-Based Gender Classification Using MFCC and Machine Learning

**Course:** EC24313 -- Artificial Intelligence and Machine Learning\
**Assessment:** 2\
**Task:** Classify voice recordings into Boys (Class 1) and Girls (Class
2) using MFCC-based acoustic features.

## Project Overview

This project implements a voice-based binary classification workflow.
Audio recordings are analyzed, converted into numerical MFCC features,
and used to train and evaluate machine-learning classifiers.

The workflow is divided into two main stages:

1.  **Audio analysis and feature extraction** -- inspect the voice
    dataset, extract MFCC features, generate visualizations, and export
    the feature table.
2.  **Model training and evaluation** -- prepare the feature table,
    train and compare SVM and Random Forest classifiers, evaluate the
    selected model, and save the final trained pipeline.

## Repository Contents

  -----------------------------------------------------------------------------
  File                                      Description
  ----------------------------------------- -----------------------------------
  `AIMLAssessment2analysis.ipynb`           Audio dataset inspection, waveform
                                            analysis, MFCC extraction, scatter
                                            plot, PCA visualization, and CSV
                                            generation.

  `AIMLAssessment2model.ipynb`              Feature validation, train-test
                                            split, model training,
                                            hyperparameter tuning, model
                                            comparison, evaluation, ROC
                                            analysis, and final model saving.

  `Final_Gender_Classification_Model.pkl`   Saved scikit-learn pipeline
                                            containing `StandardScaler` and the
                                            selected SVM classifier.

  `MFCC_Gender_Features.csv`                Extracted feature dataset
                                            containing 143 samples, 26 MFCC
                                            summary features, filename, and
                                            class label.

  `VOICE DATASET.zip`                       Original audio dataset used for
                                            analysis and MFCC feature
                                            extraction.
  -----------------------------------------------------------------------------

## Dataset

The supplied dataset contains **143 audio recordings**:

  Class label   Category      Number of recordings
  ------------- ----------- ----------------------
  1             Boys                           124
  2             Girls                           19
                **Total**                  **143**

The dataset is imbalanced, so accuracy is considered together with
class-wise precision, sensitivity (recall), balanced accuracy, macro
F1-score, and ROC-AUC.

The analysis notebook searches the extracted ZIP archive for these two
directory names (case-insensitive):

-   `VOICE DATASET BOYS`
-   `VOICE DATASET GIRLS`

Supported audio extensions in the notebook include WAV, MP3, OGG, FLAC,
M4A, AAC, and WMA.

## MFCC Feature Extraction

The analysis notebook uses **13 MFCC coefficients** for each audio
recording. For the final CSV-generation step, the mean and standard
deviation of each coefficient are calculated across the audio frames.

Therefore, each recording is represented by:

-   13 MFCC mean values
-   13 MFCC standard-deviation values
-   **26 numerical input features in total**

The CSV also contains a `Filename` column and a `Class` column.
`Filename` is not used as a model input.

> **Note:** The final model uses 26 features, not 13. The 13 refers to
> the number of MFCC coefficient sequences; their mean and standard
> deviation form the 26-dimensional feature vector.

## Machine-Learning Methods

Two classifiers are compared:

### 1. Support Vector Machine (SVM)

-   Kernel: RBF
-   Feature scaling: `StandardScaler`
-   Class weighting: `balanced`
-   Hyperparameters tuned: `C` and `gamma`
-   Selected parameters: `C = 10`, `gamma = 0.01`

### 2. Random Forest

The Random Forest classifier is used as a comparison model. Its
hyperparameter search considers the number of estimators, maximum depth,
minimum samples required to split a node, and minimum samples per leaf.
Class weighting is set to `balanced`.

Both models are tuned using `GridSearchCV` with stratified five-fold
cross-validation and **macro F1-score** as the optimization metric. The
SVM is selected because it achieves the higher cross-validation macro
F1-score.

## Reported Results

The model notebook reports the following held-out test results:

  Metric                              SVM (RBF)   Random Forest
  --------------------------------- ----------- ---------------
  Test accuracy                          0.8276          0.8621
  Balanced accuracy                      0.6900          0.5000
  Macro precision                        0.6583          0.4310
  Macro recall                           0.6900          0.5000
  Macro F1-score                         0.6712          0.4630
  Cross-validation macro F1-score        0.7374          0.5692

The SVM confusion matrix, with rows as actual classes and columns as
predicted classes in the order **Boys, Girls**, is:

  Actual class        Predicted Boys   Predicted Girls
  ----------------- ---------------- -----------------
  Boys (Class 1)                  22                 3
  Girls (Class 2)                  2                 2

For the SVM:

-   Boys sensitivity: **88%**
-   Girls sensitivity: **50%**
-   ROC-AUC for Girls as the positive class: **0.9400**

Random Forest has a slightly higher test accuracy, but it does not
correctly identify any Girls samples in the test set. The SVM is
therefore selected for its better balance between the two classes.

These values are the results reported for the notebook's held-out test
split. The saved final model is subsequently refitted using the complete
feature dataset; the test metrics should not be interpreted as a new
evaluation of that refitted model.

## Running the Notebooks

The notebooks use Google Colab upload/download utilities and `/content/`
paths. Google Colab is therefore the simplest way to reproduce the
workflow without changing those notebook cells.

### Step 1 -- Open the analysis notebook

Open `AIMLAssessment2analysis.ipynb` in Google Colab.

### Step 2 -- Run the analysis and feature extraction

Run the notebook cells in order.

-   Install/import the required libraries.
-   Upload `VOICE DATASET.zip` when prompted.
-   The notebook extracts the ZIP archive and locates the Boys and Girls
    folders.
-   It analyzes the audio files and generates waveform and feature
    visualizations.
-   It extracts the 26 MFCC summary features and exports
    `MFCC_Gender_Features.csv`.

The generated CSV can be downloaded from Colab or uploaded to the Colab
session when running the next notebook.

### Step 3 -- Open the model notebook

Open `AIMLAssessment2model.ipynb` in Google Colab.

### Step 4 -- Load the feature CSV

Upload `MFCC_Gender_Features.csv` when needed, or place it in the Colab
working directory at:

`/content/MFCC_Gender_Features.csv`

Run the notebook cells in order to validate the data, perform the
stratified 80:20 split, tune both classifiers, compare their
performance, and generate evaluation outputs.

### Step 5 -- Save the final model

The model notebook refits the selected SVM pipeline using the complete
feature dataset and saves:

`Final_Gender_Classification_Model.pkl`

The saved pipeline includes feature scaling, so a separate scaler is not
required during prediction.

## Python Libraries

The notebooks use the following main libraries:

-   `librosa` and `soundfile` -- audio loading and processing
-   `numpy` and `pandas` -- numerical operations and tabular data
-   `matplotlib` and `seaborn` -- plots and visualizations
-   `scikit-learn` -- preprocessing, model training, cross-validation,
    and evaluation
-   `joblib` -- saving the trained model

For a local Python environment, install the main dependencies with:

``` bash
pip install librosa soundfile numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

The analysis notebook also imports `google.colab.files` for file
upload/download. That import is specific to Google Colab; a local run
requires replacing the upload/download cells with local file paths.

## Using the Saved Model for Prediction

The saved model expects a 2D feature array containing **26 values per
recording**, in exactly the same order as the CSV feature columns:

1.  `MFCC1_Mean` through `MFCC13_Mean`
2.  `MFCC1_Std` through `MFCC13_Std`

Example of loading the pipeline and predicting from an already extracted
feature vector:

``` python
import joblib
import numpy as np

model = joblib.load("Final_Gender_Classification_Model.pkl")

# Replace these example values with the 26 MFCC features
# extracted from a new audio recording, in the training column order.
feature_vector = np.array([
    # 13 MFCC mean values,
    # followed by 13 MFCC standard-deviation values
], dtype=float)

prediction = model.predict(feature_vector.reshape(1, -1))[0]

if prediction == 1:
    print("Predicted class: Boys")
elif prediction == 2:
    print("Predicted class: Girls")
```

The example requires a complete 26-value feature vector. The same MFCC
extraction procedure and feature ordering used to create the CSV must be
followed for new recordings. The pipeline performs standardization
internally.

## Limitations

-   The dataset contains 143 recordings and is imbalanced between the
    two classes.
-   Only four Girls recordings are present in the held-out test set, so
    the Girls sensitivity estimate is based on a small number of
    samples.
-   Evaluation is based on one stratified train-test split.
-   Recording conditions, microphone characteristics, background noise,
    and speaker variation may affect performance.
-   Results on this dataset do not guarantee the same performance on
    unseen speakers or different recording environments.

## Suggested Future Improvements

-   Collect a larger and more balanced voice dataset.
-   Evaluate using speaker-independent train-test partitions.
-   Use repeated stratified cross-validation for more stable performance
    estimates.
-   Test additional acoustic features and classification algorithms.
-   Evaluate robustness under different recording conditions.

## Reproducibility Notes

-   The model-training notebook uses an 80:20 stratified train-test
    split with `random_state = 42`.
-   Hyperparameter selection uses stratified five-fold cross-validation
    and macro F1-score.
-   Class labels are `1 = Boys` and `2 = Girls`.
-   For ROC analysis, Class 2 (Girls) is treated as the positive class.
-   The final `.pkl` file is a trained scikit-learn pipeline and should
    be loaded with compatible versions of its dependencies.

------------------------------------------------------------------------

**Course:** EC24313 -- Artificial Intelligence and Machine Learning\
**Assessment:** 2 -- Voice-Based Gender Classification
