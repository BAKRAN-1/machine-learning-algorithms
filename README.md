# Machine Learning Algorithms

A collection of nine small Python machine-learning practice projects by Abdullah Bakran.
The examples cover regression, classification, clustering, model evaluation and hyperparameter tuning.

## Projects

| Directory | Focus |
| --- | --- |
| [gradient-descent](gradient-descent/) | Gradient descent from scratch, compared with scikit-learn on math and computer science scores. |
| [hyperparameter-tuning](hyperparameter-tuning/) | GridSearchCV and RandomizedSearchCV with Iris and handwritten digits. |
| [cross-validation](cross-validation/) | Three-fold cross-validation comparing classification models on Iris. |
| [kmeans-clustering](kmeans-clustering/) | Iris clustering, cluster centers and the elbow method. |
| [linear-regression](linear-regression/) | Linear regression on Canadian per-capita income by year. |
| [logistic-regression-binary](logistic-regression-binary/) | Employee attrition classification with salary encoding. |
| [logistic-regression-multiclass](logistic-regression-multiclass/) | Iris classification and confusion-matrix visualization. |
| [naive-bayes](naive-bayes/) | Gaussian and Multinomial Naive Bayes on the Wine dataset. |
| [support-vector-machine](support-vector-machine/) | Handwritten digit classification with RBF and linear SVM kernels. |

## Setup and running

Use Python 3.10 or newer. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
cd linear-regression
python main.py
```

Run each script from its own project directory so relative CSV paths resolve correctly.
For `hyperparameter-tuning`, `exercise.py` provides a second example on handwritten digits.
Many scripts were developed interactively: standalone expressions display in an interactive editor,
but do not print in a terminal. Matplotlib plots open in separate windows.

## Libraries and data

- NumPy and Pandas: numerical operations and tabular data.
- scikit-learn: estimators, built-in Iris/Digits/Wine datasets, cross-validation and parameter search.
- Matplotlib and Seaborn: exploratory plots and confusion matrices.
- Local CSV examples: Canadian income, test scores and employee attrition.

These are learning exercises, not deployed prediction services. Dataset ownership is not claimed;
original CSV source links were not recorded in the supplied folders.
No benchmark accuracy is claimed in this README. Some splits and estimators are randomized,
so results can vary. Hyperparameter-search scores are cross-validation scores, not an independent test evaluation.

## Packaging notes

Projects were organized into separate folders. Original desktop folders were preserved.
Environment folders and duplicate ZIP archives were excluded. The binary-classification CSV path
was updated, its group mean was restricted to numeric columns, and a duplicate Iris-label conversion
was removed from the parameter-search example.
## Validation

All 10 Python scripts across the nine projects completed successfully on Python 3.12.10 using the pinned requirements. Plots used Matplotlib's non-interactive Agg backend during this execution check. This checks script execution; it does not establish model quality or verify interactive plot appearance.
