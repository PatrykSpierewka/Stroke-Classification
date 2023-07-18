# Stroke-Classification
### Supervised learning used for stroke classification based on data from: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

The dataset contained information about various patient features, such as age, gender, marital status, occupation, glucose level, BMI (Body Mass Index), smoking status, and more. These features were used to predict the likelihood of a stroke occurrence.

The downloaded database was imported into the environment, cleared of non-numeric data, the problem of unbalanced classes was resolved by up-sampling or down-sampling, grid-search of parameters was used for most classifiers in order to obtain the best classifier parameters, records with outliers were removed, data was divided into training and test sets in the ratio of 85%:15%

### Classification carried out using the following methods:
- KNN (K-nearest-neighbours)
- SVM (Support Vector Machines)
- RF (Random Forest)
- LR (Logistic Regression)
- MLP (Multilayer Perceptron)
- NB (Naive Bayes)
- GB (Gradient Boosting)


### The performance of each classifier was evaluated using accuracy, precision, recall, and F1-score metrics. The table below presents the results:
<div align="center">

|    Classifier    |  Accuracy  | Precision |  Recall   |    F1     |
|:----------------:|:----------:|:---------:|:---------:|:---------:|
|       KNN        |    0.82    |    0.83   |    0.82   |    0.82   |
|       SVM        |    0.81    |    0.8    |    0.79   |    0.79   |
|        RF        |    0.77    |    0.76   |    0.76   |    0.76   |
|        LR        |    0.79    |    0.78   |    0.77   |    0.77   |
|       MLP        |    0.91    |    0.91   |    0.9    |    0.9    |
|        NB        |    0.77    |    0.77   |    0.77   |    0.77   |

</div>

The MLP classifier achieved the highest accuracy of 0.91, indicating its effectiveness in stroke classification.
