# TSF Task 2: Iris Dataset Classification

The Sparks Foundation (TSF) Data Science and Business Analytics internship, Task 2. The task is labeled "Prediction using Unsupervised ML," and the notebook works through the Iris dataset end to end: exploration, visualization, and classification of the three iris species.

## Dataset

`Iris.csv` — 150 rows with Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, and Species (Iris-setosa, Iris-versicolor, Iris-virginica, 50 each).

## Approach

The notebook (`TSF DS&BA TASK 2.ipynb`) works through:

1. **Exploration.** Dataset info, descriptive statistics, correlation matrix, outlier checks, a countplot of the species, and a pairplot of the four features.
2. **Preprocessing.** Label-encode the species column, take the four measurements as features, and split 80/20 (random_state=1).
3. **Modeling.** Train a `DecisionTreeClassifier(max_depth=8, random_state=1)` on the training split.
4. **Visualization.** PCA down to 2 components with a plotted decision boundary showing how the tree separates the three species, plus a countplot of the predicted classes.

## Results

Accuracy 0.9667 on the test set (30 samples). The PCA decision-boundary plot shows clean separation of the three species.

## How to run

```bash
git clone https://github.com/suhasaitham22/TSF-TASK-2-IRIS-DATASET.git
cd TSF-TASK-2-IRIS-DATASET
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook "TSF DS&BA TASK 2.ipynb"
```

The notebook expects `Iris.csv` next to it, which is already in the repo.

## Tech stack

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, Jupyter Notebook.
