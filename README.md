# PHM Wafer Fault Detection

This project addresses the fault‐detection problem posed in the HW3 data set of the Prognostic Health Management (PHM) competition. The goal is to build a predictive model that can identify failing wafer fabrication runs based on a large collection of sensor readings collected over time. Detecting failures early helps to minimise downtime and reduce maintenance costs in semiconductor manufacturing.

## Data

Three CSV files are provided:
Link:
https://www.kaggle.com/competitions/hw-3-phm

## Dataset Files

| File                  | Description                                                                                                            |
|------------------------|------------------------------------------------------------------------------------------------------------------------|
| **hw3_train.csv**      | Training data with **1,097 rows × 594 columns**. Each row = one process run. Columns include: `Id`, `Time`, ~590 sensor readings (`0–589`), and binary target **Pass/Fail** (`0 = pass`, `1 = fail`). |
| **hw3_test.csv**       | Test data with **471 rows** and the same feature columns as training, except **no target column**. Your model must predict Pass/Fail. |
| **sample_submission.csv** | Example submission file with two columns: `Id` and `Pass/Fail`. Demonstrates the required format for final predictions. |


Exploratory analysis shows that the training set is highly imbalanced, around 93 % of runs pass and only 7 % fail. Several sensor columns contain many missing values or are constant across all samples. Missing values are common in time‐series sensors and must be handled carefully.

## Evaluation metric

Competitors are ranked by the Mean Macro F1 Score, which is the unweighted average of the F1 scores for the two classes (0 and 1). In scikit‑learn this is specified via average='macro', which tells the library to calculate the F1 score for each label and take their unweighted mean
scikit-learn.org
. Unlike micro‑averaging, macro‑averaging does not take label imbalance into account, so poor performance on the minority class will strongly penalise the overall score. A naïve predictor that always outputs 0 will achieve an F1 score of 0 on the fail class and therefore a poor overall score.

## Maintenance strategies

The PHM problem relates closely to equipment maintenance policies. In traditional run‑to‑failure (also called reactive or breakdown maintenance), equipment is simply fixed when it breaks. This strategy requires little planning and can be suitable for non‑critical, low‑cost assets, but if applied universally it can lead to unplanned production halts and higher repair costs
fiixsoftware.com
.

Preventive maintenance follows a time‑ or usage‑based schedule. Assets are periodically inspected or repaired at set intervals to extend their lifespan and prevent malfunctions
fiixsoftware.com
. Modern computerised maintenance management systems (CMMS) automate work‑order scheduling, but over‑aggressive schedules can lead to unnecessary tasks and resource waste.

Predictive maintenance leverages sensor data and analytics to anticipate equipment failures. Sensors such as vibration monitors can warn of an impending malfunction so that inspections and repairs happen only when needed
fiixsoftware.com
. Predictive maintenance offers potential cost savings and deeper insights into the causes of breakdowns compared with purely preventive schedules, but requires up‑front investment in sensors and data infrastructure
fiixsoftware.com
.

## Approach

The accompanying notebook (`phm_analysis.ipynb`) details the experimentation that led to the final solution. In summary:

Feature engineering: The Time string is converted to a timestamp and calendar components (`month`, `day`, `hour`, `minute`, `weekday`). The Id column is dropped because it does not carry predictive information. All remaining columns are coerced to numeric types. No sensor columns are discarded a priori; instead, missing values are imputed.

Imputation & scaling: Missing values are filled with the median of each feature computed on the training data. Features are standardised (zero mean, unit variance) via StandardScaler. Scaling helps algorithms such as logistic regression converge and prevents features with large magnitudes from dominating the model.

Model selection: Several models were evaluated (LightGBM, XGBoost, random forests, gradient boosting). Surprisingly, a regularised logistic regression with class_weight='balanced' outperformed tree‐based methods on the macro F1 score. The linear model benefits from scaling and can handle high‐dimensional sparse data efficiently.

Threshold tuning: Because of the severe class imbalance, the default probability threshold of 0.5 leads to very low recall for the minority class. Using out‐of‐fold predictions from cross‑validation, different thresholds were tested and a value around 0.29 maximised the macro F1 score. This custom threshold is applied when converting probabilities to class labels.

## Files

`phm_model.py` – A command‑line script that implements the preprocessing, training and inference pipeline described above. It loads the provided CSV files, expands the time features, imputes and scales the data, trains a logistic regression classifier, applies the tuned threshold (0.29) and writes a submission file matching the required format.

`phm_analysis.ipynb` – A Jupyter notebook containing exploratory data analysis, model experiments and threshold tuning. It illustrates how the logistic regression model was selected and how the threshold was optimised.

`phm_submission.csv` – A ready‑to‑submit prediction file for hw3_test.csv generated by running phm_model.py. It contains two columns (Id and Pass/Fail) and predicts 6 failing runs in the test set.

## Usage

To reproduce the results or generate your own submission:

Ensure you have Python 3 with the required dependencies installed (pandas, numpy, scikit‑learn). In the competition environment these are already available.

Place hw3_train.csv and hw3_test.csv in the same directory as phm_model.py.

Run the script from the command line:

```bash
python phm_model.py --train hw3_train.csv --test hw3_test.csv --output my_submission.csv
```

This will train the logistic regression model on the full training set and write predictions for the test set to my_submission.csv.

For a deeper understanding of the data and modelling decisions, open the notebook:

jupyter notebook phm_analysis.ipynb


The notebook contains code cells that you can run interactively to reproduce the EDA, cross‑validation experiments and threshold tuning.

## Results

On the training data, the final logistic regression model achieves a cross‑validated macro F1 score of approximately 0.57, outperforming the tested tree‐based models. When applied to the test set with the tuned threshold, the model predicts six failing runs. The exact macro F1 score on the hidden leaderboard depends on the true labels of the test set, but the methodology provides a strong balance between precision and recall for both classes.
