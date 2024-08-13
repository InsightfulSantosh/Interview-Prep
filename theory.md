# Theoretical interview questions


* The list of questions is based on [this post](https://medium.com/data-science-insider/160-data-science-interview-questions-14dbd8bf0a08?source=friends_link&sk=7acf122a017c672a95f70c7cb7b585c0)
* Legend: 👶 easy ‍⭐️ medium 🚀 expert
  
## Table of contents

* [Supervised machine learning](#supervised-machinelearning)
* [Linear regression](#linear-regression)
* [Validation](#validation)
* [Classification](#classification)
* [Regularization](#regularization)
* [Feature selection](#feature-selection)
* [Decision trees](#decision-trees)
* [Random forest](#random-forest)
* [Gradient boosting](#gradient-boosting)
* [Parameter tuning](#parameter-tuning)
* [Neural networks](#neural-networks)
* [Optimization in neural networks](#optimization-in-neuralnetworks)
* [Neural networks for computer vision](#neural-networks-for-computervision)
* [Text classification](#text-classification)
* [Clustering](#clustering)
* [Dimensionality reduction](#dimensionality-reduction)
* [Ranking and search](#ranking-andsearch)
* [Recommender systems](#recommender-systems)
* [Time series](#time-series)





<br/>

## Supervised machine learning

**What is supervised machine learning? 👶**

A case when we have both features (the matrix X) and the labels (the vector y)

<br/>

## Linear regression

**What is regression? Which models can you use to solve a regression problem? 👶**

Regression is a part of supervised ML. Regression models investigate the relationship between a dependent (target) and independent variable (s) (predictor).
Here are some common regression models

- *Linear Regression* establishes a linear relationship between target and predictor (s). It predicts a numeric value and has a shape of a straight line.
- *Polynomial Regression* has a regression equation with the power of independent variable more than 1. It is a curve that fits into the data points.
- *Ridge Regression* helps when predictors are highly correlated (multicollinearity problem). It penalizes the squares of regression coefficients but doesn’t allow the coefficients to reach zeros (uses L2 regularization).
- *Lasso Regression* penalizes the absolute values of regression coefficients and allows some of the coefficients to reach absolute zero (thereby allowing feature selection).

<br/>

**What is linear regression? When do we use it? 👶**

Linear regression is a model that assumes a linear relationship between the input variables (X) and the single output variable (y).

With a simple equation:

```
y = B0 + B1*x1 + ... + Bn * xN
```

B is regression coefficients, x values are the independent (explanatory) variables  and y is dependent variable.

The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression.

Simple linear regression:

```
y = B0 + B1*x1
```

Multiple linear regression:

```
y = B0 + B1*x1 + ... + Bn * xN
```

<br/>

**What are the main assumptions of linear regression? ⭐**

The main assumptions of linear regression are crucial for the model to produce reliable and valid results. Here are the key assumptions:

1. **Linearity**  
   - The relationship between the independent variables and the dependent variable should be linear. This means that the change in the dependent variable is proportional to the change in the independent variables.

2. **Independence**  
   - The observations (data points) should be independent of each other. This means that the value of the dependent variable for one observation is not influenced by the value for another observation.

3. **Homoscedasticity**  
   - The variance of the errors (residuals) should be constant across all levels of the independent variables. In other words, the spread of the residuals should be the same for all predicted values.

4. **Normality of Residuals**  
   - The residuals (the differences between the observed and predicted values) should be normally distributed. This assumption is especially important for hypothesis testing and constructing confidence intervals.

5. **No Multicollinearity**  
   - The independent variables should not be highly correlated with each other. High multicollinearity can make it difficult to determine the individual effect of each independent variable on the dependent variable.

6. **No Autocorrelation**  
   - In time series data, the residuals should not be correlated with each other. Autocorrelation can lead to underestimated standard errors and inflated t-scores, making the model results misleading.


<br/>

**What’s the normal distribution? Why do we care about it? 👶**


The normal distribution, also known as the Gaussian distribution or bell curve, is a continuous probability distribution that is symmetrical around its mean. In a normal distribution:
- **Mean (μ)**: The central peak represents the average value of the data.
- **Standard Deviation (σ)**: Controls the spread or width of the distribution. About 68% of the data falls within one standard deviation from the mean, 95% within two standard deviations, and 99.7% within three standard deviations.

Mathematically, the probability density function (PDF) of a normal distribution is given by:

$$
f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} \left(\frac{x-\mu}{\sigma}\right)^2}
$$

**Why Do We Care About the Normal Distribution?**

The normal distribution is fundamental in statistics for several reasons:

1. **Central Limit Theorem (CLT)**:
   - The CLT states that the sum or average of a large number of independent and identically distributed random variables tends to follow a normal distribution, regardless of the original distribution. This makes the normal distribution a cornerstone for many statistical methods.

2. **Hypothesis Testing**:
   - Many statistical tests, like t-tests and z-tests, assume that the data follows a normal distribution. This assumption allows for the derivation of critical values and p-values, which are used to make inferences about populations.

3. **Confidence Intervals**:
   - When data is normally distributed, confidence intervals for the mean can be easily constructed, providing a range within which the true population mean is likely to fall.

4. **Real-World Phenomena**:
   - Many natural phenomena, like heights, weights, and test scores, tend to follow a normal distribution, making it a useful model for real-world data.

5. **Simplicity and Symmetry**:
   - The symmetry and simplicity of the normal distribution make it mathematically tractable and a useful model for various statistical analyses, even when data only approximates normality.


<br/>

**How do we check if a variable follows the normal distribution? ‍⭐️**

1. Plot a histogram out of the sampled data. If you can fit the bell-shaped "normal" curve to the histogram, then the hypothesis that the underlying random variable follows the normal distribution can not be rejected.
2. Check Skewness and Kurtosis of the sampled data. Skewness = 0 and kurtosis = 3 are typical for a normal distribution, so the farther away they are from these values, the more non-normal the distribution.
3. Use Kolmogorov-Smirnov or/and Shapiro-Wilk tests for normality. They take into account both Skewness and Kurtosis simultaneously.
4. Check for Quantile-Quantile plot. It is a scatterplot created by plotting two sets of quantiles against one another. Normal Q-Q plot place the data points in a roughly straight line.

<br/>

**What if we want to build a model for predicting prices? Are prices distributed normally? Do we need to do any pre-processing for prices? ‍⭐️**

Data is not normal. Specially, real-world datasets or uncleaned datasets always have certain skewness. Same goes for the price prediction. Price of houses or any other thing under consideration depends on a number of factors. So, there's a great chance of presence of some skewed values i.e outliers if we talk in data science terms. 

Yes, you may need to do pre-processing. Most probably, you will need to remove the outliers to make your distribution near-to-normal.

<br/>

**What methods for solving linear regression do you know? ‍⭐️**

To solve linear regression, you need to find the coefficients , which minimize the sum of squared errors.


"There are several methods to solve linear regression, each with its own applications and considerations:

1. **Ordinary Least Squares (OLS)**
   - **Description**: OLS is the most common method for solving linear regression. It minimizes the sum of the squared differences between the observed and predicted values.
   - **Use Case**: OLS is suitable when the assumptions of linear regression are met, particularly when the data is well-behaved, and there is no multicollinearity.

2. **Gradient Descent**
   - **Description**: Gradient Descent is an iterative optimization algorithm used to minimize the cost function. It is particularly useful when dealing with large datasets or when the number of features is very high.
   - **Use Case**: It's often used when OLS is computationally expensive or infeasible, such as in high-dimensional spaces or with large datasets.

 3. **Singular Value Decomposition (SVD)**
   - **Description**: SVD is a factorization method that decomposes the design matrix into singular vectors and singular values. It’s useful for solving linear regression in cases where the matrix is not full-rank or when multicollinearity is present.
   - **Use Case**: SVD is employed in situations where the design matrix is ill-conditioned or when dimensionality reduction is needed.

4. **Ridge Regression**
   - **Description**: Ridge regression is a regularized version of OLS that adds a penalty term proportional to the square of the coefficients. This helps prevent overfitting, especially in the presence of multicollinearity.
   - **Use Case**: Used when multicollinearity exists among the independent variables, as it can help stabilize the solution.

5. **Lasso Regression**
   - **Description**: Lasso regression adds an L1 regularization term to the cost function, which can shrink some coefficients to zero, effectively performing feature selection.
   - **Use Case**: Ideal for situations where feature selection is important, or when you suspect that only a subset of the predictors are significant.

6. **Normal Equation**
   - **Description**: The normal equation provides a closed-form solution for linear regression by directly solving the matrix equation derived from the OLS method.
   - **Use Case**: It’s effective for small to medium-sized datasets where matrix inversion is computationally feasible.

7. **Least Absolute Deviations (LAD)**
   - **Description**: LAD minimizes the sum of the absolute deviations between the observed and predicted values. It is robust to outliers compared to OLS.
   - **Use Case**: Used in cases where the data contains significant outliers that could affect the OLS solution.

In practice, the choice of method depends on the dataset’s characteristics, including its size, the presence of multicollinearity, and whether feature selection is needed."


<br/>

**What is gradient descent? How does it work? ‍⭐️**

Gradient descent is an algorithm that uses calculus concept of gradient to try and reach local or global minima. It works by taking the negative of the gradient in a point of a given function, and updating that point repeatedly using the calculated negative gradient, until the algorithm reaches a local or global minimum, which will cause future iterations of the algorithm to return values that are equal or too close to the current point. It is widely used in machine learning applications.

<br/>

**What is the normal equation? ‍⭐️**


The normal equation provides a closed-form solution to the linear regression problem. It directly computes the coefficients that minimize the cost function (the sum of squared residuals) without requiring iterative optimization techniques like gradient descent.

The Normal Equation Formula

For a linear regression model $\hat{y} = X\beta$, where:
- $X$ is the matrix of input features (including a column of ones for the intercept),
- $\beta$ is the vector of coefficients we want to estimate,
- $\hat{y}$ is the vector of predicted values.

The normal equation is given by:

$$
\beta = (X^T X)^{-1} X^T y
$$

Explanation

- $X^T$ is the transpose of the feature matrix $X$,
- $(X^T X)^{-1}$ is the inverse of the matrix $X^T X$,
- $y$ is the vector of observed values.

How It Works

- The normal equation minimizes the sum of squared differences between the observed values and the values predicted by the model.
- By multiplying the transpose of $X$ by $X$, we obtain a square matrix that can be inverted (assuming it's not singular). This inverse, when multiplied by $X^T y$, gives us the coefficient vector $\beta$.

Use Case

- The normal equation is particularly useful for small to medium-sized datasets where the matrix inversion is computationally feasible. However, it can become inefficient or impractical for very large datasets or when the matrix $X^T X$ is singular or nearly singular, leading to numerical instability.

Summary

The normal equation provides an efficient way to directly compute the optimal coefficients for linear regression, making it an important tool in situations where an exact, non-iterative solution is desired.


<br/>

**What is SGD  —  stochastic gradient descent? What’s the difference with the usual gradient descent? ‍⭐️**

In both gradient descent (GD) and stochastic gradient descent (SGD), you update a set of parameters in an iterative manner to minimize an error function.

While in GD, you have to run through ALL the samples in your training set to do a single update for a parameter in a particular iteration, in SGD, on the other hand, you use ONLY ONE or SUBSET of training sample from your training set to do the update for a parameter in a particular iteration. If you use SUBSET, it is called Minibatch Stochastic gradient Descent.

<br/>

**Which metrics for evaluating regression models do you know? 👶**

1. Mean Squared Error(MSE)
2. Root Mean Squared Error(RMSE)
3. Mean Absolute Error(MAE)
4. R² or Coefficient of Determination
5. Adjusted R²

<br/>

**What are MSE and RMSE? 👶**

MSE stands for <strong>M</strong>ean <strong>S</strong>quare <strong>E</strong>rror while RMSE stands for <strong>R</strong>oot <strong>M</strong>ean <strong>S</strong>quare <strong>E</strong>rror. They are metrics with which we can evaluate models.

<br/>

**What is the bias-variance trade-off? 👶**

**Bias** is the error introduced by approximating the true underlying function, which can be quite complex, by a simpler model. **Variance** is a model sensitivity to changes in the training dataset.

**Bias-variance trade-off** is a relationship between the expected test error and the variance and the bias - both contribute to the level of the test error and ideally should be as small as possible:

```
ExpectedTestError = Variance + Bias² + IrreducibleError
```

But as a model complexity increases, the bias decreases and the variance increases which leads to *overfitting*. And vice versa, model simplification helps to decrease the variance but it increases the bias which leads to *underfitting*.

<br/>


## Validation

**What is overfitting? 👶**

When your model perform very well on your training set but can't generalize the test set, because it adjusted a lot to the training set.

<br/>

**How to validate your models? 👶**

One of the most common approaches is splitting data into train, validation and test parts.
Models are trained on train data, hyperparameters (for example early stopping) are selected based on the validation data, the final measurement is done on test dataset.
Another approach is cross-validation: split dataset into K folds and each time train models on training folds and measure the performance on the validation folds.
Also you could combine these approaches: make a test/holdout dataset and do cross-validation on the rest of the data. The final quality is measured on test dataset.

<br/>

**Why do we need to split our data into three parts: train, validation, and test? 👶**

The training set is used to fit the model, i.e. to train the model with the data. The validation set is then used to provide an unbiased evaluation of a model while fine-tuning hyperparameters. This improves the generalization of the model. Finally, a test data set which the model has never "seen" before should be used for the final evaluation of the model. This allows for an unbiased evaluation of the model. The evaluation should never be performed on the same data that is used for training. Otherwise the model performance would not be representative.

<br/>

**Can you explain how cross-validation works? 👶**

Cross-validation is the process to separate your total training set into two subsets: training and validation set, and evaluate your model to choose the hyperparameters. But you do this process iteratively, selecting differents training and validation set, in order to reduce the bias that you would have by selecting only one validation set.

<br/>

**What is K-fold cross-validation? 👶**

K fold cross validation is a method of cross validation where we select a hyperparameter k. The dataset is now divided into k parts. Now, we take the 1st part as validation set and remaining k-1 as training set. Then we take the 2nd part as validation set and remaining k-1 parts as training set. Like this, each part is used as validation set once and the remaining k-1 parts are taken together and used as training set.
It should not be used in a time series data.

<br/>

**How do we choose K in K-fold cross-validation? What’s your favorite K? 👶**

There are two things to consider while deciding K: the number of models we get and the size of validation set. We do not want the number of models to be too less, like 2 or 3. At least 4 models give a less biased decision on the metrics. On the other hand, we would want the dataset to be at least 20-25% of the entire data. So that at least a ratio of 3:1 between training and validation set is maintained. <br/>
I tend to use 4 for small datasets and 5 for large ones as K.

<br/>


## Classification

**What is classification? Which models would you use to solve a classification problem? 👶**

Classification problems are problems in which our prediction space is discrete, i.e. there is a finite number of values the output variable can be. Some models which can be used to solve classification problems are: logistic regression, decision tree, random forests, multi-layer perceptron, one-vs-all, amongst others.

<br/>

**What is logistic regression? When do we need to use it? 👶**

Logistic regression is a Machine Learning algorithm that is used for binary classification. You should use logistic regression when your Y variable takes only two values, e.g. True and False, "spam" and "not spam", "churn" and "not churn" and so on. The variable is said to be a "binary" or "dichotomous".

<br/>

**Is logistic regression a linear model? Why? 👶**

Yes, Logistic Regression is considered a generalized linear model because the outcome always depends on the sum of the inputs and parameters. Or in other words, the output cannot depend on the product (or quotient, etc.) of its parameters.

<br/>

**What is sigmoid? What does it do? 👶**

A sigmoid function is a type of activation function, and more specifically defined as a squashing function. Squashing functions limit the output to a range between 0 and 1, making these functions useful in the prediction of probabilities.

Sigmod(x) = 1/(1+e^{-x})

<br/>

**How do we evaluate classification models? 👶**

Depending on the classification problem, we can use the following evaluation metrics:

1. Accuracy
2. Precision
3. Recall
4. F1 Score
5. Logistic loss (also known as Cross-entropy loss)
6. Jaccard similarity coefficient score

<br/>

**What is accuracy? 👶**

Accuracy is a metric for evaluating classification models. It is calculated by dividing the number of correct predictions by the number of total predictions.

<br/>

**Is accuracy always a good metric? 👶**

Accuracy is not a good performance metric when there is imbalance in the dataset. For example, in binary classification with 95% of A class and 5% of B class, a constant prediction of A class would have an accuracy of 95%. In case of imbalance dataset, we need to choose Precision, recall, or F1 Score depending on the problem we are trying to solve.

<br/>

**What is the confusion table? What are the cells in this table? 👶**

Confusion table (or confusion matrix) shows how many True positives (TP), True Negative (TN), False Positive (FP) and False Negative (FN) model has made.

||                |     Actual   |        Actual |
|:---:|   :---:        |     :---:    |:---:          |
||                | Positive (1) | Negative (0)  |
|Predicted|   Positive (1) | TP           | FP            |
|Predicted|   Negative (0) | FN           | TN            |

* True Positives (TP): When the actual class of the observation is 1 (True) and the prediction is 1 (True)
* True Negative (TN): When the actual class of the observation is 0 (False) and the prediction is 0 (False)
* False Positive (FP): When the actual class of the observation is 0 (False) and the prediction is 1 (True)
* False Negative (FN): When the actual class of the observation is 1 (True) and the prediction is 0 (False)

Most of the performance metrics for classification models are based on the values of the confusion matrix.

<br/>

**What are precision, recall, and F1-score? 👶**

* Precision and recall are classification evaluation metrics:
* P = TP / (TP + FP) and R = TP / (TP + FN).
* Where TP is true positives, FP is false positives and FN is false negatives
* In both cases the score of 1 is the best: we get no false positives or false negatives and only true positives.
* F1 is a combination of both precision and recall in one score (harmonic mean):
* F1 = 2 * PR / (P + R).
* Max F score is 1 and min is 0, with 1 being the best.

<br/>

**Precision-recall trade-off ‍⭐️**

Tradeoff means increasing one parameter would lead to decreasing of other. Precision-recall tradeoff occur due to increasing one of the parameter(precision or recall) while keeping the model same. 

In an ideal scenario where there is a perfectly separable data, both precision and recall can get maximum value of 1.0. But in most of the practical situations, there is noise in the dataset and the dataset is not perfectly separable. There might be some points of positive class closer to the negative class and vice versa. In such cases, shifting the decision boundary can either increase the precision or recall but not both. Increasing one parameter leads to decreasing of the other. 

<br/>

**What is the ROC curve? When to use it? ‍⭐️**

ROC stands for *Receiver Operating Characteristics*. The diagrammatic representation that shows the contrast between true positive rate vs true negative rate. It is used when we need to predict the probability of the binary outcome.

<br/>

**What is AUC (AU ROC)? When to use it? ‍⭐️**

AUC stands for *Area Under the ROC Curve*. ROC is a probability curve and AUC represents degree or measure of separability. It's used when we need to value how much model is capable of distinguishing between classes.  The value is between 0 and 1, the higher the better.

<br/>

**How to interpret the AU ROC score? ‍⭐️**

1. **Definition**:
   - The AU ROC score, or AUC-ROC, measures the area under the Receiver Operating Characteristic (ROC) curve.

2. **Range**:
   - The AUC score ranges from 0 to 1.

3. **Interpretation**:
   - **AUC = 1**: Perfect model; the model can perfectly distinguish between positive and negative classes.
   - **0.5 < AUC < 1**: Model performs better than random guessing; higher values indicate better performance.
   - **AUC = 0.5**: Model performs no better than random guessing; it’s as if predictions are made by chance.
   - **AUC < 0.5**: Model performs worse than random guessing; the model might be predicting in the reverse direction.

4. **Purpose**:
   - Measures the overall ability of the model to discriminate between the positive and negative classes.

5. **Use Cases**:
   - Useful for comparing the performance of different models.
   - Effective for evaluating models on imbalanced datasets.

6. **Threshold Independence**:
   - AUC considers performance across all possible classification thresholds, not just a single threshold.

7. **Practical Application**:
   - A higher AUC score indicates a better-performing model, making it a key metric for model evaluation and selection.

An excellent model has AUC near to the 1 which means it has good measure of separability. A poor model has AUC near to the 0 which means it has worst measure of separability. When AUC score is 0.5, it means model has no class separation capacity whatsoever. 

<br/>

**What is the PR (precision-recall) curve? ‍⭐️**

A *precision*-*recall curve* (or PR Curve) is a plot of the precision (y-axis) and the recall (x-axis) for different probability thresholds. Precision-recall curves (PR curves) are recommended for highly skewed domains where ROC curves may provide an excessively optimistic view of the performance.

<br/>

**What is the area under the PR curve? Is it a useful metric? ‍⭐️I**

The Precision-Recall AUC is just like the ROC AUC, in that it summarizes the curve with a range of threshold values as a single score.

A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate.

<br/>

**In which cases AU PR is better than AU ROC? ‍⭐️**

What is different however is that AU ROC looks at a true positive rate TPR and false positive rate FPR while AU PR looks at positive predictive value PPV and true positive rate TPR.

Typically, if true negatives are not meaningful to the problem or you care more about the positive class, AU PR is typically going to be more useful; otherwise, If you care equally about the positive and negative class or your dataset is quite balanced, then going with AU ROC is a good idea.

<br/>

**What do we do with categorical variables? ‍⭐️**

Categorical variables must be encoded before they can be used as features to train a machine learning model. There are various encoding techniques, including:
- One-hot encoding
- Label encoding
- Ordinal encoding
- Target encoding

<br/>

**Why do we need one-hot encoding? ‍⭐️**

If we simply encode categorical variables with a Label encoder, they become ordinal which can lead to undesirable consequences. In this case, linear models will treat category with id 4 as twice better than a category with id 2. One-hot encoding allows us to represent a categorical variable in a numerical vector space which ensures that vectors of each category have equal distances between each other. This approach is not suited for all situations, because by using it with categorical variables of high cardinality (e.g. customer id) we will encounter problems that come into play because of the curse of dimensionality.

<br/>


## Regularization

**What happens to our linear regression model if we have three columns in our data: x, y, z  —  and z is a sum of x and y? ‍⭐️**

We would not be able to perform the resgression. Because z is linearly dependent on x and y so when performing the regression <img src="https://render.githubusercontent.com/render/math?math={X}^{T}{X}"> would be a singular (not invertible) matrix.
<br/>

**What happens to our linear regression model if the column z in the data is a sum of columns x and y and some random noise? ‍⭐️**

Answer here

<br/>

**What is regularization? Why do we need it? 👶**

Regularization is used to reduce overfitting in machine learning models. It helps the models to generalize well and make them robust to outliers and noise in the data.

<br/>

**Which regularization techniques do you know? ‍⭐️**

There are mainly two types of regularization,
1. L1 Regularization (Lasso regularization) - Adds the sum of absolute values of the coefficients to the cost function. <img src="https://render.githubusercontent.com/render/math?math=\lambda\sum_{i=1}^{n} \left | w_i \right |">
2. L2 Regularization (Ridge regularization) - Adds the sum of squares of coefficients to the cost function. <img src="https://render.githubusercontent.com/render/math?math=\lambda\sum_{i=1}^{n} {w_{i}}^{2}">

* Where <img src="https://render.githubusercontent.com/render/math?math=\lambda"> determines the amount of regularization.

<br/>

**What kind of regularization techniques are applicable to linear models? ‍⭐️**

AIC/BIC, Ridge regression, Lasso, Elastic Net, Basis pursuit denoising, Rudin–Osher–Fatemi model (TV), Potts model, RLAD,
Dantzig Selector,SLOPE

<br/>

**How does L2 regularization look like in a linear model? ‍⭐️**

L2 regularization adds a penalty term to our cost function which is equal to the sum of squares of models coefficients multiplied by a lambda hyperparameter. This technique makes sure that the coefficients are close to zero and is widely used in cases when we have a lot of features that might correlate with each other.

<br/>

**How do we select the right regularization parameters? 👶**

Regularization parameters can be chosen using a grid search, for example https://scikit-learn.org/stable/modules/linear_model.html has one formula for the implementing for regularization, alpha in the formula mentioned can be found by doing a RandomSearch or a GridSearch on a set of values and selecting the alpha which gives the least cross validation or validation error.


<br/>

**What’s the effect of L2 regularization on the weights of a linear model? ‍⭐️**

L2 regularization penalizes larger weights more severely (due to the squared penalty term), which encourages weight values to decay toward zero.

<br/>

**How L1 regularization looks like in a linear model? ‍⭐️**

L1 regularization adds a penalty term to our cost function which is equal to the sum of modules of models coefficients multiplied by a lambda hyperparameter. For example, cost function with L1 regularization will look like: <img src="https://render.githubusercontent.com/render/math?math=\sum_{i=0}^{N}%20(y_i%20-%20\sum_{j=0}^{M}%20x_{ij}%20*%20w_j)%2B\lambda\sum_{j=0}^{M}%20\left%20|%20w_j%20\right%20|">

<br/>

**What’s the difference between L2 and L1 regularization? ‍⭐️**

- Penalty terms: L1 regularization uses the sum of the absolute values of the weights, while L2 regularization uses the sum of the weights squared.
- Feature selection: L1 performs feature selection by reducing the coefficients of some predictors to 0, while L2 does not.
- Computational efficiency: L2 has an analytical solution, while L1 does not.
- Multicollinearity: L2 addresses multicollinearity by constraining the coefficient norm.

<br/>

**Can we have both L1 and L2 regularization components in a linear model? ‍⭐️**

Yes, elastic net regularization combines L1 and L2 regularization. 

<br/>

**What’s the interpretation of the bias term in linear models? ‍⭐️**

Bias is simply, a difference between predicted value and actual/true value. It can be interpreted as the distance from the average prediction and true value i.e. true value minus mean(predictions). But dont get confused between accuracy and bias.

<br/>

**How do we interpret weights in linear models? ‍⭐️**

Without normalizing weights or variables, if you increase the corresponding predictor by one unit, the coefficient represents on average how much the output changes. By the way, this interpretation still works for logistic regression - if you increase the corresponding predictor by one unit, the weight represents the change in the log of the odds.

If the variables are normalized, we can interpret weights in linear models like the importance of this variable in the predicted result.

<br/>

**If a weight for one variable is higher than for another  —  can we say that this variable is more important? ‍⭐️**

Yes - if your predictor variables are normalized.

Without normalization, the weight represents the change in the output per unit change in the predictor. If you have a predictor with a huge range and scale that is used to predict an output with a very small range - for example, using each nation's GDP to predict maternal mortality rates - your coefficient should be very small. That does not necessarily mean that this predictor variable is not important compared to the others.

<br/>

**When do we need to perform feature normalization for linear models? When it’s okay not to do it? ‍⭐️**

Feature normalization is necessary for L1 and L2 regularizations. The idea of both methods is to penalize all the features relatively equally. This can't be done effectively if every feature is scaled differently. 

Linear regression without regularization techniques can be used without feature normalization. Also, regularization can help to make the analytical solution more stable, — it adds the regularization matrix to the feature matrix before inverting it. 

<br/>


## Feature selection

**What is feature selection? Why do we need it? 👶**

Feature Selection is a method used to select the relevant features for the model to train on. We need feature selection to remove the irrelevant features which leads the model to under-perform.  

<br/>

**Is feature selection important for linear models? ‍⭐️**

Answer here

<br/>

**Which feature selection techniques do you know? ‍⭐️**

Here are some of the feature selections:
- Principal Component Analysis
- Neighborhood Component Analysis
- ReliefF Algorithm

<br/>

**Can we use L1 regularization for feature selection? ‍⭐️**

Yes, because the nature of L1 regularization will lead to sparse coefficients of features. Feature selection can be done by keeping only features with non-zero coefficients.

<br/>

**Can we use L2 regularization for feature selection? ‍⭐️**

Answer here

<br/>


## Decision trees

**What are the decision trees? 👶**

This is a type of supervised learning algorithm that is mostly used for classification problems. Surprisingly, it works for both categorical and continuous dependent variables. 

In this algorithm, we split the population into two or more homogeneous sets. This is done based on most significant attributes/ independent variables to make as distinct groups as possible.

A decision tree is a flowchart-like tree structure, where each internal node (non-leaf node) denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (or terminal node) holds a value for the target variable.

Various techniques : like Gini, Information Gain, Chi-square, entropy.

<br/>

**How do we train decision trees? ‍⭐️**

1. Start at the root node.
2. For each variable X, find the set S_1 that minimizes the sum of the node impurities in the two child nodes and choose the split {X*,S*} that gives the minimum over all X and S.
3. If a stopping criterion is reached, exit. Otherwise, apply step 2 to each child node in turn.

<br/>

**What are the main parameters of the decision tree model? 👶**

* maximum tree depth
* minimum samples per leaf node
* impurity criterion

<br/>

**How do we handle categorical variables in decision trees? ‍⭐️**

Some decision tree algorithms can handle categorical variables out of the box, others cannot. However, we can transform categorical variables, e.g. with a binary or a one-hot encoder.

Handling categorical variables in decision trees involves a few key steps. Here’s a point-by-point guide on how to handle categorical variables when using decision trees:


- **Direct Handling**: Many decision tree algorithms handle categorical variables directly.
- **One-Hot Encoding**: Convert categories into binary columns.
- **Label Encoding**: Assign integers to categories (less common for decision trees).
- **Frequency Encoding**: Replace categories with their frequency.
- **Target Encoding**: Replace categories with the mean of the target variable.
- **High Cardinality**: Group less frequent categories.
- **Interaction Features**: Create features that capture interactions between categorical variables.
- **Handling Missing Values**: Impute missing values before encoding.

Using these methods ensures that categorical variables are appropriately represented in decision tree models, leading to better performance and interpretability.

<br/>

**What are the benefits of a single decision tree compared to more complex models? ‍⭐️**

* easy to implement
* fast training
* fast inference
* good explainability

<br/>

**How can we know which features are more important for the decision tree model? ‍⭐️**

Often, we want to find a split such that it minimizes the sum of the node impurities. The impurity criterion is a parameter of decision trees. Popular methods to measure the impurity are the Gini impurity and the entropy describing the information gain.

<br/>


## Random forest

**What is random forest? 👶**

Random Forest is a machine learning method for regression and classification which is composed of many decision trees. Random Forest belongs to a larger class of ML algorithms called ensemble methods (in other words, it involves the combination of several models to solve a single prediction problem).

<br/>

**Why do we need randomization in random forest? ‍⭐️**

Random forest in an extention of the **bagging** algorithm which takes *random data samples from the training dataset* (with replacement), trains several models and averages predictions. In addition to that, each time a split in a tree is considered, random forest takes a *random sample of m features from full set of n features* (with replacement) and uses this subset of features as candidates for the split (for example, `m = sqrt(n)`).

Training decision trees on random data samples from the training dataset *reduces variance*. Sampling features for each split in a decision tree *decorrelates trees*.

<br/>

**What are the main parameters of the random forest model? ‍⭐️**

- `max_depth`: Longest Path between root node and the leaf
- `min_sample_split`: The minimum number of observations needed to split a given node
- `max_leaf_nodes`: Conditions the splitting of the tree and hence, limits the growth of the trees
- `min_samples_leaf`: minimum number of samples in the leaf node
- `n_estimators`: Number of trees
- `max_sample`: Fraction of original dataset given to any individual tree in the given model
- `max_features`: Limits the maximum number of features provided to trees in random forest model

<br/>

**How do we select the depth of the trees in random forest? ‍⭐️**

The greater the depth, the greater amount of information is extracted from the tree, however, there is a limit to this, and the algorithm even if defensive against overfitting may learn complex features of noise present in data and as a result, may overfit on noise. Hence, there is no hard thumb rule in deciding the depth, but literature suggests a few tips on tuning the depth of the tree to prevent overfitting:

- limit the maximum depth of a tree
- limit the number of test nodes
- limit the minimum number of objects at a node required to split
- do not split a node when, at least, one of the resulting subsample sizes is below a given threshold
- stop developing a node if it does not sufficiently improve the fit.

<br/>

**How do we know how many trees we need in random forest? ‍⭐️**

The number of trees in random forest is worked by n_estimators, and a random forest reduces overfitting by increasing the number of trees. There is no fixed thumb rule to decide the number of trees in a random forest, it is rather fine tuned with the data, typically starting off by taking the square of the number of features (n) present in the data followed by tuning until we get the optimal results.

<br/>

**Is it easy to parallelize training of a random forest model? How can we do it? ‍⭐️**

Yes, R provides a simple way to parallelize training of random forests on large scale data.
It makes use of a parameter called multicombine which can be set to TRUE for parallelizing random forest computations.

```R
rf <- foreach(ntree=rep(25000, 6), .combine=randomForest::combine,
              .multicombine=TRUE, .packages='randomForest') %dopar% {
    randomForest(x, y, ntree=ntree)
}
```


<br/>

**What are the potential problems with many large trees? ‍⭐️**

Answer here

<br/>

**What if instead of finding the best split, we randomly select a few splits and just select the best from them. Will it work? 🚀**

Answer here

<br/>

**What happens when we have correlated features in our data? ‍⭐️**

In random forest, since random forest samples some features to build each tree, the information contained in correlated features is twice as much likely to be picked than any other information contained in other features. 

In general, when you are adding correlated features, it means that they linearly contains the same information and thus it will reduce the robustness of your model. Each time you train your model, your model might pick one feature or the other to "do the same job" i.e. explain some variance, reduce entropy, etc.

<br/>


## Gradient boosting

**What is gradient boosting trees? ‍⭐️**

Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.

<br/>

**What’s the difference between random forest and gradient boosting? ‍⭐️**

   1. Random Forests builds each tree independently while Gradient Boosting builds one tree at a time.
   2. Random Forests combine results at the end of the process (by averaging or "majority rules") while Gradient Boosting combines     results along the way.

<br/>

**Is it possible to parallelize training of a gradient boosting model? How to do it? ‍⭐️**

Yes, different frameworks provide different options to make training faster, using GPUs to speed up the process by making it highly parallelizable.For example, for XGBoost <i>tree_method = 'gpu_hist'</i> option makes training faster by use of GPUs. 

<br/>

**Feature importance in gradient boosting trees  —  what are possible options? ‍⭐️**

Answer here

<br/>

**Are there any differences between continuous and discrete variables when it comes to feature importance of gradient boosting models? 🚀**

Answer here

<br/>

**What are the main parameters in the gradient boosting model? ‍⭐️**

There are many parameters, but below are a few key defaults.
* learning_rate=0.1 (shrinkage).
* n_estimators=100 (number of trees).
* max_depth=3.
* min_samples_split=2.
* min_samples_leaf=1.
* subsample=1.0.

<br/>

**How do you approach tuning parameters in XGBoost or LightGBM? 🚀**

Depending upon the dataset, parameter tuning can be done manually or using hyperparameter optimization frameworks such as optuna and hyperopt. In manual parameter tuning, we need to be aware of max-depth, min_samples_leaf and min_samples_split so that our model does not overfit the data but try to predict generalized characteristics of data (basically keeping variance and bias low for our model).

<br/>

**How do you select the number of trees in the gradient boosting model? ‍⭐️**

Most implementations of gradient boosting are configured by default with a relatively small number of trees, such as hundreds or thousands. Using scikit-learn we can perform a grid search of the n_estimators model parameter

<br/>



## Parameter tuning

**Which hyper-parameter tuning strategies (in general) do you know? ‍⭐️**

There are several strategies for hyper-tuning but I would argue that the three most popular nowadays are the following:
* <b>Grid Search</b> is an exhaustive approach such that for each hyper-parameter, the user needs to <i>manually</i> give a list of values for the algorithm to try. After these values are selected, grid search then evaluates the algorithm using each and every combination of hyper-parameters and returns the combination that gives the optimal result (i.e. lowest MAE). Because grid search evaluates the given algorithm using all combinations, it's easy to see that this can be quite computationally expensive and can lead to sub-optimal results specifically since the user needs to specify specific values for these hyper-parameters, which is prone for error and requires domain knowledge.

* <b>Random Search</b> is similar to grid search but differs in the sense that rather than specifying which values to try for each hyper-parameter, an upper and lower bound of values for each hyper-parameter is given instead. With uniform probability, random values within these bounds are then chosen and similarly, the best combination is returned to the user. Although this seems less intuitive, no domain knowledge is necessary and theoretically much more of the parameter space can be explored.

* In a completely different framework, <b>Bayesian Optimization</b> is thought of as a more statistical way of optimization and is commonly used when using neural networks, specifically since one evaluation of a neural network can be computationally costly. In numerous research papers, this method heavily outperforms Grid Search and Random Search and is currently used on the Google Cloud Platform as well as AWS. Because an in-depth explanation requires a heavy background in bayesian statistics and gaussian processes (and maybe even some game theory), a "simple" explanation is that a much simpler/faster <i>acquisition function</i> intelligently chooses (using a <i>surrogate function</i> such as probability of improvement or GP-UCB) which hyper-parameter values to try on the computationally expensive, original algorithm. Using the result of the initial combination of values on the expensive/original function, the acquisition function takes the result of the expensive/original algorithm into account and uses it as its prior knowledge to again come up with another set of hyper-parameters to choose during the next iteration. This process continues either for a specified number of iterations or for a specified amount of time and similarly the combination of hyper-parameters that performs the best on the expensive/original algorithm is chosen.


<br/>

**What’s the difference between grid search parameter tuning strategy and random search? When to use one or another? ‍⭐️**

For specifics, refer to the above answer.

<br/>


## Neural networks

**What kind of problems neural nets can solve? 👶**

Neural nets are good at solving non-linear problems. Some good examples are problems that are relatively easy for humans (because of experience, intuition, understanding, etc), but difficult for traditional regression models: speech recognition, handwriting recognition, image identification, etc.

<br/>

**How does a usual fully-connected feed-forward neural network work? ‍⭐️**

In a usual fully-connected feed-forward network, each neuron receives input from every element of the previous layer and thus the receptive field of a neuron is the entire previous layer. They are usually used to represent feature vectors for input data in classification problems but can be expensive to train because of the number of computations involved.

<br/>

**Why do we need activation functions? 👶**

The main idea of using neural networks is to learn complex nonlinear functions. If we are not using an activation function in between different layers of a neural network, we are just stacking up multiple linear layers one on top of another and this leads to learning a linear function. The Nonlinearity comes only with the activation function, this is the reason we need activation functions.

<br/>

**What are the problems with sigmoid as an activation function? ‍⭐️**

The derivative of the sigmoid function for large positive or negative numbers is almost zero. From this comes the problem of vanishing gradient — during the backpropagation our net will not learn (or will learn drastically slow). One possible way to solve this problem is to use ReLU activation function.

<br/>

**What is ReLU? How is it better than sigmoid or tanh? ‍⭐️**

ReLU is an abbreviation for Rectified Linear Unit. It is an activation function which has the value 0 for all negative values and the value f(x) = x for all positive values. The ReLU has a simple activation function which makes it fast to compute and while the sigmoid and tanh activation functions saturate at higher values, the ReLU has a potentially infinite activation, which addresses the problem of vanishing gradients. 

<br/>

**How we can initialize the weights of a neural network? ‍⭐️**

Proper initialization of weight matrix in neural network is very necessary.
Simply we can say there are two ways for initializtions.
   1. Initializing weights with zeroes.
      Setting weights to zero makes your network no better than a linear model. It is important to note that setting biases to 0 will not create any troubles as non zero weights take care of breaking the symmetry and even if bias is 0, the values in every neuron are still different.  
   2. Initializing weights randomly.
      Assigning random values to weights is better than just 0 assignment. 
* a) If weights are initialized with very high values the term np.dot(W,X)+b becomes significantly higher and if an activation function like sigmoid() is applied, the function maps its value near to 1 where the slope of gradient changes slowly and learning takes a lot of time.
* b) If weights are initialized with low values it gets mapped to 0, where the case is the same as above. This problem is often referred to as the vanishing gradient.
      
<br/>

**What if we set all the weights of a neural network to 0? ‍⭐️**

If all the weights of a neural network are set to zero, the output of each connection is same (W*x = 0). This means the gradients which are backpropagated to each connection in a layer is same. This means all the connections/weights learn the same thing, and the model never converges. 

<br/>

**What regularization techniques for neural nets do you know? ‍⭐️**

* L1 Regularization - Defined as the sum of absolute values of the individual parameters. The L1 penalty causes a subset of the weights to become zero, suggesting that the corresponding features may safely be discarded. 
* L2 Regularization - Defined as the sum of square of individual parameters. Often supported by regularization hyperparameter alpha. It results in weight decay. 
* Data Augmentation - This requires some fake data to be created as a part of training set. 
* Drop Out : This is most effective regularization technique for newral nets. Few randome nodes in each layer is deactivated in forward pass. This allows the algorithm to train on different set of nodes in each iterations.
<br/>

**What is dropout? Why is it useful? How does it work? ‍⭐️**

Dropout is a technique that at each training step turns off each neuron with a certain probability of *p*. This way at each iteration we train only *1-p* of neurons, which forces the network not to rely only on the subset of neurons for feature representation. This leads to regularizing effects that are controlled by the hyperparameter *p*.  

<br/>


## Optimization in neural networks

**What is backpropagation? How does it work? Why do we need it? ‍⭐️**

The Backpropagation algorithm looks for the minimum value of the error function in weight space using a technique called the delta rule or gradient descent. 
The weights that minimize the error function is then considered to be a solution to the learning problem. 

We need backpropogation because,
* Calculate the error – How far is your model output from the actual output.
* Minimum Error – Check whether the error is minimized or not.
* Update the parameters – If the error is huge then, update the parameters (weights and biases). After that again check the error.  
Repeat the process until the error becomes minimum.
* Model is ready to make a prediction – Once the error becomes minimum, you can feed some inputs to your model and it will produce the output.

<br/>

**Which optimization techniques for training neural nets do you know? ‍⭐️**

* Gradient Descent
* Stochastic Gradient Descent
* Mini-Batch Gradient Descent(best among gradient descents)
* Nesterov Accelerated Gradient
* Momentum
* Adagrad 
* AdaDelta
* Adam(best one. less time, more efficient)

<br/>

**How do we use SGD (stochastic gradient descent) for training a neural net? ‍⭐️**

SGD approximates the expectation with few randomly selected samples (instead of the full data). In comparison to batch gradient descent, we can efficiently approximate the expectation in large data sets using SGD. For neural networks this reduces the training time a lot even considering that it will converge later as the random sampling adds noise to the gradient descent.

<br/>

**What’s the learning rate? 👶**

The learning rate is an important hyperparameter that controls how quickly the model is adapted to the problem during the training. It can be seen as the "step width" during the parameter updates, i.e. how far the weights are moved into the direction of the minimum of our optimization problem.

<br/>

**What happens when the learning rate is too large? Too small? 👶**

A large learning rate can accelerate the training. However, it is possible that we "shoot" too far and miss the minimum of the function that we want to optimize, which will not result in the best solution. On the other hand, training with a small learning rate takes more time but it is possible to find a more precise minimum. The downside can be that the solution is stuck in a local minimum, and the weights won't update even if it is not the best possible global solution.

<br/>

**How to set the learning rate? ‍⭐️**

There is no straightforward way of finding an optimum learning rate for a model. It involves a lot of hit and trial. Usually starting with a small values such as 0.01 is a good starting point for setting a learning rate and further tweaking it so that it doesn't overshoot or converge too slowly.

<br/>

**What is Adam? What’s the main difference between Adam and SGD? ‍⭐️**

Adam (Adaptive Moment Estimation) is a optimization technique for training neural networks. on an average, it is the best optimizer .It works with momentums of first and second order. The intuition behind the Adam is that we don’t want to roll so fast just because we can jump over the minimum, we want to decrease the velocity a little bit for a careful search.

Adam tends to converge faster, while SGD often converges to more optimal solutions.
SGD's high variance disadvantages gets rectified by Adam (as advantage for Adam).

<br/>

**When would you use Adam and when SGD? ‍⭐️**

Adam tends to converge faster, while SGD often converges to more optimal solutions.

<br/>

**Do we want to have a constant learning rate or we better change it throughout training? ‍⭐️**

Answer here

<br/>

**How do we decide when to stop training a neural net? 👶**

Simply stop training when the validation error is the minimum.

<br/>

**What is model checkpointing? ‍⭐️**

Saving the weights learned by a model mid training for long running processes is known as model checkpointing so that you can resume your training from a certain checkpoint.

<br/>

**Can you tell us how you approach the model training process? ‍⭐️**

Answer here

<br/>


## Neural networks for computer vision

**How we can use neural nets for computer vision? ‍⭐️**

Neural nets used in the area of computer vision are generally Convolutional Neural Networks(CNN's). You can learn about convolutions below. It appears that convolutions are quite powerful when it comes to working with images and videos due to their ability to extract and learn complex features. Thus CNN's are a go-to method for any problem in computer vision.    

<br/>

**What’s a convolutional layer? ‍⭐️**

The idea of the convolutional layer is the assumption that the information needed for making a decision often is spatially close and thus, it only takes the weighted sum over nearby inputs. It also assumes that the networks’ kernels can be reused for all nodes, hence the number of weights can be drastically reduced. To counteract only one feature being learnt per layer, multiple kernels are applied to the input which creates parallel channels in the output. Consecutive layers can also be stacked to allow the network to find more high-level features.

<br/>

**Why do we actually need convolutions? Can’t we use fully-connected layers for that? ‍⭐️**

A fully-connected layer needs one weight per inter-layer connection, which means the number of weights which needs to be computed quickly balloons as the number of layers and nodes per layer is increased. 

<br/>

**What’s pooling in CNN? Why do we need it? ‍⭐️**

Pooling is a technique to downsample the feature map. It allows layers which receive relatively undistorted versions of the input to learn low level features such as lines, while layers deeper in the model can learn more abstract features such as texture.

<br/>

**How does max pooling work? Are there other pooling techniques? ‍⭐️**

Max pooling is a technique where the maximum value of a receptive field is passed on in the next feature map. The most commonly used receptive field is 2 x 2 with a stride of 2, which means the feature map is downsampled from N x N to N/2 x N/2. Receptive fields larger than 3 x 3 are rarely employed as too much information is lost. 

Other pooling techniques include:

* Average pooling, the output is the average value of the receptive field.
* Min pooling, the output is the minimum value of the receptive field.
* Global pooling, where the receptive field is set to be equal to the input size, this means the output is equal to a scalar and can be used to reduce the dimensionality of the feature map. 

<br/>

**Are CNNs resistant to rotations? What happens to the predictions of a CNN if an image is rotated? 🚀**

CNNs are not resistant to rotation by design. However, we can make our models resistant by augmenting our datasets with different rotations of the raw data. The predictions of a CNN will change if an image is rotated and we did not augment our dataset accordingly. A demonstration of this occurence can be seen in [this video](https://www.youtube.com/watch?v=VO1bQo4PXV4), where a CNN changes its predicted class between a duck and a rabbit based on the rotation of the image.

<br/>

**What are augmentations? Why do we need them? 👶**

Augmentations are an artifical way of expanding the existing datasets by performing some transformations, color shifts or many other things on the data. It helps in diversifying the data and even increasing the data when there is scarcity of data for a model to train on.  

<br/>

**What kind of augmentations do you know? 👶**

There are many kinds of augmentations which can be used according to the type of data you are working on some of which are geometric and numerical transformation, PCA, cropping, padding, shifting, noise injection etc.

<br/>

**How to choose which augmentations to use? ‍⭐️**

Augmentations really depend on the type of output classes and the features you want your model to learn. For eg. if you have mostly properly illuminated images in your dataset and want your model to predict poorly illuminated images too, you can apply channel shifting on your data and include the resultant images in your dataset for better results.

<br/>

**What kind of CNN architectures for classification do you know? 🚀**

Image Classification
* Inception v3
* Xception 
* DenseNet
* AlexNet
* VGG16
* ResNet
* SqueezeNet
* EfficientNet
* MobileNet

The last three are designed so they use smaller number of parameters which is helpful for edge AI. 

<br/>

**What is transfer learning? How does it work? ‍⭐️**

Given a source domain D_S and learning task T_S, a target domain D_T and learning task T_T, transfer learning aims to help improve the learning of the target predictive function f_T in D_T using the knowledge in D_S and T_S, where D_S ≠ D_T,or T_S ≠ T_T. In other words, transfer learning enables to reuse knowledge coming from other domains or learning tasks.

In the context of CNNs, we can use networks that were pre-trained on popular datasets such as ImageNet. We then can use the weights of the layers that learn to represent features and combine them with a new set of layers that learns to map the feature representations to the given classes. Two popular strategies are either to freeze the layers that learn the feature representations completely, or to give them a smaller learning rate.

<br/>

**What is object detection? Do you know any architectures for that? 🚀**

Answer here

<br/>

**What is object segmentation? Do you know any architectures for that? 🚀**

Answer here

<br/>


## Text classification

**How can we use machine learning for text classification? ‍⭐️**

Answer here

<br/>

**What is bag of words? How we can use it for text classification? ‍⭐️**

Bag of Words is a representation of text that describes the occurrence of words within a document. The order or structure of the words is not considered. For text classification, we look at the histogram of the words within the text and consider each word count as a feature.

<br/>

**What are the advantages and disadvantages of bag of words? ‍⭐️**

Advantages:
1. Simple to understand and implement.

Disadvantages:
1. The vocabulary requires careful design, most specifically in order to manage the size, which impacts the sparsity of the document representations.
2. Sparse representations are harder to model both for computational reasons (space and time complexity) and also for information reasons
3. Discarding word order ignores the context, and in turn meaning of words in the document. Context and meaning can offer a lot to the model, that if modeled could tell the difference between the same words differently arranged (“this is interesting” vs “is this interesting”), synonyms (“old bike” vs “used bike”).

<br/>

**What are N-grams? How can we use them? ‍⭐️**

The function to tokenize into consecutive sequences of words is called n-grams. It can be used to find out N most co-occurring words (how often word X is followed by word Y) in a given sentence.

<br/>

**How large should be N for our bag of words when using N-grams? ‍⭐️**

When using N-grams in a bag-of-words model, the choice of $( N )$ depends on:

- **Task Complexity**: Smaller $( N )$ (e.g., 1 or 2) is often sufficient for simple tasks like sentiment analysis. Larger $( N )$ (e.g., 3 or 4) captures more context but increases computational complexity.
- **Computational Resources**: Larger $( N )$ values can lead to a very sparse feature matrix and higher computational costs.
- **Sparsity and Overfitting**: Larger $( N )$ may cause overfitting and high sparsity, especially with smaller datasets.

**In summary**: Start with $( N = 1 )$ or $( N = 2 )$ and adjust based on the specific task and dataset, balancing context capture with computational feasibility.

<br/>

**What is TF-IDF? How is it useful for text classification? ‍⭐️**

Term Frequency (TF) is a scoring of the frequency of the word in the current document. Inverse Document Frequency(IDF) is a scoring of how rare the word is across documents. It is used in scenario where highly recurring words may not contain as much informational content as the domain specific words. For example, words like “the” that are frequent across all documents therefore need to be less weighted. The TF-IDF score highlights words that are distinct (contain useful information) in a given document.  

<br/>

**Which model would you use for text classification with bag of words features? ‍⭐️**
For text classification using bag-of-words features, several models are commonly used, each with its own strengths:

1. **Logistic Regression**:
   - **Why**: Simple, efficient, and performs well for text classification tasks. It works well with sparse features like those from a bag-of-words model.
   
2. **Naive Bayes (Multinomial or Bernoulli)**:
   - **Why**: Particularly effective for text classification, especially with word frequency features. It assumes feature independence, which simplifies the model and makes it scalable.

3. **Support Vector Machine (SVM)**:
   - **Why**: Effective for high-dimensional data and can handle large feature spaces. SVMs with linear kernels work well with bag-of-words features.

4. **Random Forest**:
   - **Why**: Provides robustness and handles a variety of features. It's less sensitive to overfitting compared to other models but can be more computationally intensive.

5. **Gradient Boosting Machines (e.g., XGBoost, LightGBM)**:
   - **Why**: Often provides high accuracy and can handle complex patterns in data. They are powerful for large datasets and diverse feature sets.


**In summary**: **Naive Bayes** and **Logistic Regression** are commonly used and well-suited for text classification with bag-of-words features due to their simplicity and effectiveness. **SVM** can also be a strong choice if dealing with high-dimensional data.

<br/>

**Would you prefer gradient boosting trees model or logistic regression when doing text classification with bag of words? ‍⭐️**

Usually logistic regression is better because bag of words creates a matrix with large number of columns. For a huge number of columns logistic regression is usually faster than gradient boosting trees.

<br/>

**What are word embeddings? Why are they useful? Do you know Word2Vec? ‍⭐️**

**Word Embeddings**:

- **Definition**: Word embeddings are a type of word representation that captures the semantic meaning of words in a dense vector format. Unlike traditional one-hot encoding, which represents words as high-dimensional, sparse vectors, embeddings map words into a lower-dimensional continuous vector space.

- **Purpose**: They encode semantic relationships and contextual information about words, allowing the model to understand meanings and similarities between words based on their usage in context.

**Why They Are Useful**:

1. **Semantic Understanding**: Embeddings capture the meanings of words and their relationships, enabling models to understand synonyms and word similarities.

2. **Dimensionality Reduction**: They transform high-dimensional, sparse representations into dense, lower-dimensional vectors, improving computational efficiency and performance.

3. **Contextual Relationships**: Embeddings can represent words with similar meanings close together in vector space, making them effective for capturing context and relationships in language.

4. **Transfer Learning**: Pre-trained embeddings can be used across various NLP tasks, providing a good starting point and reducing the need for extensive training on specific tasks.

**Word2Vec**:

- **Definition**: Word2Vec is a popular word embedding technique developed by Google. It uses neural networks to learn word representations from large corpora. Word2Vec offers two main models:
  - **Continuous Bag of Words (CBOW)**: Predicts a target word based on its surrounding context words.
  - **Skip-gram**: Predicts context words given a target word.

- **Features**:
  - **Contextual Learning**: Captures word meanings by training on the context in which words appear.
  - **Pre-trained Models**: Word2Vec has pre-trained models available, which can be directly used in various NLP applications.

In summary, word embeddings are crucial for capturing semantic meanings and relationships in text, improving model performance and efficiency. Word2Vec is a foundational technique for generating these embeddings and remains widely used in NLP applications.

<br/>

**Do you know any other ways to get word embeddings? 🚀**

Yes, besides Word2Vec, several other techniques are commonly used to obtain word embeddings. Here are some of the most popular ones:

1. **GloVe (Global Vectors for Word Representation)**:
   - **Definition**: GloVe is an unsupervised learning algorithm for generating word embeddings based on word co-occurrence statistics from a corpus.
   - **Key Feature**: It creates embeddings by factorizing the word co-occurrence matrix and captures global statistical information.

2. **FastText**:
   - **Definition**: Developed by Facebook's AI Research (FAIR) lab, FastText extends Word2Vec by representing each word as a bag of character n-grams, which helps in capturing subword information.
   - **Key Feature**: Useful for handling out-of-vocabulary words and morphologically rich languages.

3. **ELMo (Embeddings from Language Models)**:
   - **Definition**: ELMo generates contextualized word embeddings using deep, bidirectional language models.
   - **Key Feature**: Provides embeddings that vary depending on the context in which a word appears, offering richer and more flexible representations.

4. **BERT (Bidirectional Encoder Representations from Transformers)**:
   - **Definition**: BERT is a transformer-based model that provides contextual embeddings by training on bidirectional context.
   - **Key Feature**: Captures deep contextual meanings and relationships between words, improving performance on various NLP tasks.

5. **GPT (Generative Pre-trained Transformer)**:
   - **Definition**: GPT models generate word embeddings based on a unidirectional transformer architecture.
   - **Key Feature**: Focuses on generative language modeling and can provide powerful embeddings for text generation tasks.

6. **Sentence Transformers (e.g., SBERT)**:
   - **Definition**: Extends BERT to generate sentence-level embeddings, which can be used to derive meaningful embeddings for larger text units.
   - **Key Feature**: Useful for tasks requiring sentence or document-level understanding.

These methods offer different advantages depending on the complexity of the text and the specific requirements of the task.

<br/>

**If you have a sentence with multiple words, you may need to combine multiple word embeddings into one. How would you do it? ‍⭐️**

Approaches ranked from simple to more complex:

1. Take an average over all words
2. Take a weighted average over all words. Weighting can be done by inverse document frequency (idf part of tf-idf).
3. Use ML model like LSTM or Transformer.

<br/>

**Would you prefer gradient boosting trees model or logistic regression when doing text classification with embeddings? ‍⭐️**

**In summary:** Begin with Logistic Regression for simplicity and efficiency. If higher accuracy is needed or if the data exhibits complex patterns, consider using Gradient Boosting Trees to leverage their advanced capabilities.

<br/>

**How can you use neural nets for text classification? 🚀**

Answer here

<br/>

**How can we use CNN for text classification? 🚀**

Answer here

<br/>


## Clustering

**What is unsupervised learning? 👶**

Unsupervised learning aims to detect paterns in data where no labels are given.

<br/>

**What is clustering? When do we need it? 👶**

Clustering algorithms group objects such that similar feature points are put into the same groups (clusters) and dissimilar feature points are put into different clusters.

<br/>

**Do you know how K-means works? ‍⭐️**

1. Partition points into k subsets.
2. Compute the seed points as the new centroids of the clusters of the current partitioning.
3. Assign each point to the cluster with the nearest seed point.
4. Go back to step 2 or stop when the assignment does not change.

<br/>

**How to select K for K-means? ‍⭐️**

* Domain knowledge, i.e. an expert knows the value of k
* Elbow method: compute the clusters for different values of k, for each k, calculate the total within-cluster sum of square, plot the sum according to the number of clusters and use the band as the number of clusters.
* Average silhouette method: compute the clusters for different values of k, for each k, calculate the average silhouette of observations, plot the silhouette according to the number of clusters and select the maximum as the number of clusters.

<br/>

**What are the other clustering algorithms do you know? ‍⭐️**

* k-medoids: Takes the most central point instead of the mean value as the center of the cluster. This makes it more robust to noise.
* Agglomerative Hierarchical Clustering (AHC): hierarchical clusters combining the nearest clusters starting with each point as its own cluster.
* DIvisive ANAlysis Clustering (DIANA): hierarchical clustering starting with one cluster containing all points and splitting the clusters until each point describes its own cluster.
* Density-Based Spatial Clustering of Applications with Noise (DBSCAN): Cluster defined as maximum set of density-connected points.

<br/>

**Do you know how DBScan works? ‍⭐️**

* Two input parameters epsilon (neighborhood radius) and minPts (minimum number of points in an epsilon-neighborhood)
* Cluster defined as maximum set of density-connected points.
* Points p_j and p_i are density-connected w.r.t. epsilon and minPts if there is a point o such that both, i and j are density-reachable from o w.r.t. epsilon and minPts.
* p_j is density-reachable from p_i w.r.t. epsilon, minPts if there is a chain of points p_i -> p_i+1 -> p_i+x = p_j such that p_i+x is directly density-reachable from p_i+x-1.
* p_j is a directly density-reachable point of the neighborhood of p_i if dist(p_i,p_j) <= epsilon.

<br/>

**When would you choose K-means and when DBScan? ‍⭐️**

* DBScan is more robust to noise.
* DBScan is better when the amount of clusters is difficult to guess.
* K-means has a lower complexity, i.e. it will be much faster, especially with a larger amount of points.

<br/>


## Dimensionality reduction
**What is the curse of dimensionality? Why do we care about it? ‍⭐️**

Data in only one dimension is relatively tightly packed. Adding a dimension stretches the points across that dimension, pushing them further apart. Additional dimensions spread the data even further making high dimensional data extremely sparse. We care about it, because it is difficult to use machine learning in sparse spaces.

<br/>

**Do you know any dimensionality reduction techniques? ‍⭐️**

* Singular Value Decomposition (SVD)
* Principal Component Analysis (PCA)
* Linear Discriminant Analysis (LDA)
* T-distributed Stochastic Neighbor Embedding (t-SNE)
* Autoencoders
* Fourier and Wavelet Transforms

<br/>

**What’s singular value decomposition? How is it typically used for machine learning? ‍⭐️**

* Singular Value Decomposition (SVD) is a general matrix decomposition method that factors a matrix X into three matrices L (left singular values), Σ (diagonal matrix) and R^T (right singular values).
* For machine learning, Principal Component Analysis (PCA) is typically used. It is a special type of SVD where the singular values correspond to the eigenvectors and the values of the diagonal matrix are the squares of the eigenvalues. We use these features as they are statistically descriptive.
* Having calculated the eigenvectors and eigenvalues, we can use the Kaiser-Guttman criterion, a scree plot or the proportion of explained variance to determine the principal components (i.e. the final dimensionality) that are useful for dimensionality reduction.

<br/>


## Ranking and search

**What is the ranking problem? Which models can you use to solve them? ‍⭐️**

The ranking problem involves ordering items or documents based on their relevance or importance relative to a query or context. It is common in applications like search engines, recommendation systems, and personalized content delivery. The goal is to predict the order in which items should be presented to users.

### Key Aspects of the Ranking Problem

1. **Objective**: To sort a list of items so that the most relevant or preferred items appear first.
2. **Input**: A set of items and associated features, often in response to a user query or context.
3. **Output**: An ordered list of items.

### Models for Solving Ranking Problems

- **Pointwise Models**: Handle individual item relevance (e.g., Logistic Regression).
- **Pairwise Models**: Focus on item comparisons (e.g., RankNet, RankSVM).
- **Listwise Models**: Optimize the order of an entire list (e.g., LambdaMART, ListNet).
- **Learning-to-Rank Models**: Specialized models for ranking (e.g., XGBoost for Ranking, BERT for Ranking).

Choosing the appropriate model depends on the specific requirements of the ranking task and the nature of the data.

<br/>

**What are good unsupervised baselines for text information retrieval? ‍⭐️**

For text information retrieval, unsupervised baselines are useful to establish baseline performance without relying on labeled data. Here are some effective unsupervised baselines:

- **Bag-of-Words (BoW)**: Simple and effective for initial baselines.
- **TF-IDF**: Improves on BoW by considering term importance.
- **Latent Semantic Analysis (LSA)**: Captures latent semantic structures.
- **Latent Dirichlet Allocation (LDA)**: Provides topic modeling.
- **Word Embeddings (Word2Vec, GloVe)**: Captures semantic relationships.
- **Doc2Vec**: Represents entire documents as vectors.
- **Latent Semantic Indexing (LSI)**: Reduces dimensionality with SVD.
- **BM25**: Advanced probabilistic retrieval model.

These unsupervised baselines provide a starting point for evaluating text information retrieval performance without requiring labeled training data.

<br/>

**How would you evaluate your ranking algorithms? Which offline metrics would you use? ‍⭐️**

Evaluating ranking algorithms involves assessing how well the model orders items according to their relevance or importance. For offline evaluation, several metrics are commonly used to measure ranking quality:

- **MAP**: Average precision across queries.
- **NDCG**: Considers relevance and position with discounting.
- **P@K**: Precision in the top-K positions.
- **R@K**: Recall in the top-K positions.
- **MRR**: Average rank of the first relevant item.
- **Hit Rate**: Proportion of queries with relevant items in top-K.
- **CG**: Total relevance of items in the top-K positions.

These metrics help evaluate the effectiveness of ranking algorithms by assessing how well they prioritize relevant items and handle the ordering of results.

<br/>

**What is precision and recall at k? ‍⭐️**

Precision at k and recall at k are evaluation metrics for ranking algorithms. Precision at k shows the share of relevant items in the first *k* results of the ranking algorithm. And Recall at k indicates the share of relevant items returned in top *k* results out of all correct answers for a given query.

Example:
For a search query "Car" there are 3 relevant products in your shop. Your search algorithm returns 2 of those relevant products in the first 5 search results.
Precision at 5 = # num of relevant products in search result / k = 2/5 = 40%
Recall at 5 = # num of relevant products in search result / # num of all relevant products = 2/3 = 66.6%

<br/>

**What is mean average precision at k? ‍⭐️**

**Mean Average Precision at K (MAP@K)** is a metric used to evaluate the quality of a ranking system by measuring how well the top-K results match the relevance of the items.

### Definition

- **Average Precision at K (AP@K)**: For a single query, it calculates the average precision of the relevant items in the top-K positions of the ranked list. Precision is calculated at each relevant item, and then these precision values are averaged.
- **Mean Average Precision at K (MAP@K)**: It extends the AP@K metric to multiple queries by averaging the AP@K scores across all queries.

Formula

1. **Precision at K (P@K)**:
   - Measures the proportion of relevant items in the top-K results.
   - Formula:
     $$
     P@K = \frac{\text{Number of relevant items in top K}}{K}
     $$

2. **Average Precision at K (AP@K)**:
   - Calculates precision for each relevant item in the top-K positions and averages these values.
   - Formula:
     $$
     AP@K = \frac{1}{\text{Number of relevant items in top K}} \sum_{i=1}^{K} \left( \text{Precision at } i \times \text{Relevance}(i) \right)
     $$

3. **Mean Average Precision at K (MAP@K)**:
   - Averages the AP@K scores over all queries.
   - Formula:
     $$
     MAP@K = \frac{1}{Q} \sum_{q=1}^{Q} AP@K_q
     $$
   where $( Q )$ is the total number of queries, and $( AP@K_q )$ is the average precision at K for query $( q )$.

### Use Case

- **Evaluation**: MAP@K is useful for evaluating ranking systems, particularly when comparing different models or tuning hyperparameters. It provides a comprehensive measure of the ranking quality by considering both precision and the position of relevant items.


<br/>

**How can we use machine learning for search? ‍⭐️**

Machine learning can significantly enhance search systems by improving the relevance, accuracy, and efficiency of search results. Here’s how machine learning can be applied to various aspects of search:

1. **Query Understanding**
   - **Intent Detection**: Machine learning models can analyze queries to determine user intent, which helps in delivering more relevant results.
   - **Natural Language Processing (NLP)**: Techniques like named entity recognition (NER) and part-of-speech tagging can help in understanding the context and entities within a query.

2. **Ranking and Relevance**
   - **Learning-to-Rank**: Algorithms such as RankNet, LambdaMART, and RankSVM can be used to train models that learn to order search results based on relevance.
   - **Gradient Boosting Trees**: Models like XGBoost or LightGBM can be used to optimize the ranking of search results by considering various features and their interactions.
   - **Neural Networks**: Deep learning models, including transformers like BERT and GPT, can be used to better capture semantic relationships and improve result relevance.

3. **Personalization**
   - **User Profiling**: Machine learning models can build profiles based on user behavior, preferences, and interactions to deliver personalized search results.
   - **Collaborative Filtering**: Techniques can be used to recommend items based on similar users' behavior and preferences.

4. **Query Expansion**
   - **Synonym Expansion**: Machine learning can identify synonyms and related terms to expand the query and improve search coverage.
   - **Contextual Expansion**: Using embeddings to understand the context of queries and suggest additional terms or phrases.

5. **Document Classification and Tagging**
   - **Automatic Categorization**: Machine learning models can classify documents into predefined categories or tags to enhance search accuracy.
   - **Topic Modeling**: Techniques like LDA (Latent Dirichlet Allocation) can be used to identify and group documents by topics.

6. **Anomaly Detection**
   - **Outlier Detection**: Machine learning can help detect unusual search patterns or queries, which may indicate issues such as spam or fraudulent activity.

7. **Query Recommendation**
   - **Autocomplete and Suggestions**: Predictive models can suggest queries or corrections based on user input and historical data.
   - **Search Intent Prediction**: Models can predict the likely queries users might want to follow up on, improving their search experience.

8. **Image and Voice Search**
   - **Image Recognition**: Machine learning models can be used to index and search images based on their content.
   - **Voice Search**: Speech recognition models can convert spoken queries into text and process them to deliver relevant results.

<br/>

**How can we get training data for our ranking algorithms? ‍⭐️**

To get training data for ranking algorithms, we can:

1. **Collect User Interaction Data**: Use click-through data, purchase history, and session logs to infer relevance.
2. **Gather Explicit Feedback**: Use user ratings, reviews, or surveys to get direct relevance labels.
3. **Analyze Implicit Feedback**: Track time on page, bounce rates, and CTR to gauge relevance.
4. **Use Query Logs**: Leverage search query logs to see which results users engage with.
5. **Manual Annotation**: Label relevance manually through expert or crowdsourced annotation.
6. **Utilize Public Datasets**: Use datasets like MS MARCO or LETOR for training.
7. **Generate Synthetic Data**: Simulate user interactions or augment data to increase sample size.

These methods help create a robust dataset for training effective ranking models.

<br/>

**Can we formulate the search problem as a classification problem? How? ‍⭐️**

Yes, we can formulate the search problem as a classification problem by treating the relevance of each document or item with respect to a query as a binary classification task.

How It Works:

1. **Binary Classification**:
   - **Positive Class**: Label documents that are relevant to the query as the positive class (e.g., 1).
   - **Negative Class**: Label documents that are irrelevant as the negative class (e.g., 0).
   - **Model Training**: Train a binary classifier (e.g., logistic regression, SVM) to predict whether a document is relevant or not based on features such as query-document similarity, metadata, and user behavior.

2. **Multi-class Classification**:
   - If relevance is on a graded scale (e.g., "not relevant," "somewhat relevant," "highly relevant"), we can extend the problem to multi-class classification where each class represents a different level of relevance.

3. **Rank-as-Classification**:
   - Assign ranks to results by classifying them into bins or categories representing different relevance levels. This is often done in multi-class settings.

This approach allows us to leverage classification models to improve search relevance.

<br/>

**How can we use clicks data as the training data for ranking algorithms? 🚀**

We can use click data as training data for ranking algorithms by leveraging the implicit feedback it provides about user preferences. Here's how:

- **Pairwise Ranking**: Use clicked vs. non-clicked pairs to learn ranking preferences.
- **Pointwise Ranking**: Assign relevance scores to clicks and train models to predict these scores.
- **Listwise Ranking**: Optimize the entire list based on click patterns.
- **Bias Handling**: Adjust for position bias and examine skip behavior to improve model accuracy.

Using click data allows us to leverage real user interactions to train more effective ranking algorithms.

<br/>

**Do you know how to use gradient boosting trees for ranking? 🚀**

Yes, gradient boosting trees can be used for ranking tasks through an approach called **Learning to Rank (LTR)**. Here's how it works:

1. **Understanding the Approach**
   - **Learning to Rank**: This involves training models to predict the order of items (e.g., documents, products) rather than just predicting individual scores. Gradient boosting is particularly well-suited for this task because it can capture complex patterns in the data.

2. **Popular Gradient Boosting Frameworks**
   - **XGBoost, LightGBM, and CatBoost**: These frameworks have built-in support for ranking tasks, specifically tailored for learning-to-rank scenarios.

3. **Types of Ranking Approaches**:
   - **Pointwise**: Treat the ranking problem as a regression or classification problem, predicting a relevance score for each item. The items are then ranked based on these scores.
   - **Pairwise**: Train the model to predict the relative order between pairs of items. The goal is to minimize the number of inversions (where a less relevant item is ranked higher than a more relevant one).
   - **Listwise**: Optimize the ranking of an entire list of items at once, considering the entire list's order.

4. **How to Implement with Gradient Boosting**:
   - **XGBoost**: Use the `rank:pairwise`, `rank:ndcg`, or `rank:map` objectives for ranking tasks. These options directly optimize for pairwise ranking loss, NDCG, or MAP, respectively.
   - **LightGBM**: Specify the `lambdarank` objective, which is based on the LambdaRank algorithm and is designed for ranking tasks.
   - **CatBoost**: Similar to LightGBM, CatBoost supports ranking with the `YetiRank` objective, which optimizes the ranking order of items.

5. **Training Process**:
   - **Data Preparation**: Prepare your dataset with features that describe the items and include a relevance label (e.g., a click or rating).
   - **Group Data**: When training, group the data by query or session to ensure the model learns to rank items within each group correctly.
   - **Model Training**: Train the model using the selected gradient boosting framework and the appropriate ranking objective.


Gradient boosting trees are effective for ranking because they can handle complex interactions in features and directly optimize for ranking-specific metrics.

<br/>

**How do you do an online evaluation of a new ranking algorithm? ‍⭐️**

Online evaluation of a new ranking algorithm involves testing the algorithm in a real-world setting with live user interactions. Here's how you can conduct it:

1. **A/B Testing**
   - **Control and Treatment Groups**: Divide users into two groups: one using the current ranking algorithm (control) and the other using the new algorithm (treatment).
   - **Metrics**: Measure key performance indicators (KPIs) such as click-through rate (CTR), conversion rate, dwell time, and user engagement for both groups.
   - **Comparison**: Compare the performance of the new algorithm against the baseline (control) to determine if it provides a significant improvement.

2. **Interleaving**
   - **Interleaved Results**: Instead of splitting users, combine results from both the old and new algorithms into a single ranked list. Present this list to users.
   - **User Preference**: Observe which results users click on more often to determine which algorithm ranks items more effectively.
   - **Quick Feedback**: Interleaving allows for faster feedback and avoids the need to split traffic between different algorithms.

3. **Multi-Armed Bandit**
   - **Dynamic Allocation**: Use a multi-armed bandit approach to dynamically allocate more traffic to the better-performing algorithm while still exploring the potential of the new one.
   - **Adaptiveness**: This method adapts over time, directing more users to the algorithm that shows better performance, thus optimizing user experience while still evaluating the new model.

4. **Monitoring User Behavior**
   - **User Feedback**: Collect explicit feedback from users through surveys or ratings on the relevance of the results provided by the new algorithm.
   - **Behavioral Analysis**: Track user behavior metrics such as the number of queries per session, time spent on site, bounce rates, and repeat visits.

5. **Rolling Deployment (Canary Testing)**
   - **Gradual Rollout**: Deploy the new algorithm to a small percentage of users initially and monitor its performance closely.
   - **Expand Gradually**: If the new algorithm performs well, gradually increase the percentage of users until it is fully deployed.

6. **Key Performance Indicators (KPIs)**
   - **Engagement Metrics**: CTR, conversion rate, and time spent on results.
   - **Satisfaction Metrics**: User satisfaction scores, survey feedback.
   - **Business Metrics**: Revenue per user, overall sales, or other domain-specific metrics.



Online evaluation allows for real-time feedback and ensures the new ranking algorithm positively impacts user experience and business outcomes before full deployment.

<br/>


## Recommender systems

**What is a recommender system? 👶**

Recommender systems are software tools and techniques that provide suggestions for items that are most likely of interest to a particular user.

<br/>

**What are good baselines when building a recommender system? ‍⭐️**

* A good recommer system should give relevant and personalized information.
* It should not recommend items the user knows well or finds easily.
* It should make diverse suggestions.
* A user should explore new items.

<br/>

**What is collaborative filtering? ‍⭐️**

* Collaborative filtering is the most prominent approach to generate recommendations.
* It uses the wisdom of the crowd, i.e. it gives recommendations based on the experience of others.
* A recommendation is calculated as the average of other experiences.
* Say we want to give a score that indicates how much user u will like an item i. Then we can calculate it with the experience of N other users U as r_ui = 1/N * sum(v in U) r_vi.
* In order to rate similar experiences with a higher weight, we can introduce a similarity between users that we use as a multiplier for each rating.
* Also, as users have an individual profile, one user may have an average rating much larger than another user, so we use normalization techniques (e.g. centering or Z-score normalization) to remove the users' biases.
* Collaborative filtering does only need a rating matrix as input and improves over time. However, it does not work well on sparse data, does not work for cold starts (see below) and usually tends to overfit.

<br/>

**How we can incorporate implicit feedback (clicks, etc) into our recommender systems? ‍⭐️**

In comparison to explicit feedback, implicit feedback datasets lack negative examples. For example, explicit feedback can be a positive or a negative rating, but implicit feedback may be the number of purchases or clicks. One popular approach to solve this problem is named weighted alternating least squares (wALS) [Hu, Y., Koren, Y., & Volinsky, C. (2008, December). Collaborative filtering for implicit feedback datasets. In Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on (pp. 263-272). IEEE.]. Instead of modeling the rating matrix directly, the numbers (e.g. amount of clicks) describe the strength in observations of user actions. The model tries to find latent factors that can be used to predict the expected preference of a user for an item.

<br/>

**What is the cold start problem? ‍⭐️**

Collaborative filterung incorporates crowd knowledge to give recommendations for certain items. Say we want to recommend how much a user will like an item, we then will calculate the score using the recommendations of other users for this certain item. We can distinguish between two different ways of a cold start problem now. First, if there is a new item that has not been rated yet, we cannot give any recommendation. Also, when there is a new user, we cannot calculate a similarity to any other user.

<br/>

**Possible approaches to solving the cold start problem? ‍⭐️🚀**

* Content-based filtering incorporates features about items to calculate a similarity between them. In this way, we can recommend items that have a high similarity to items that a user liked already. In this way, we are not dependant on the ratings of other users for a given item anymore and solve the cold start problem for new items.
* Demographic filtering incorporates user profiles to calculate a similarity between them and solves the cold start problem for new users.

<br/>


## Time series

**What is a time series? 👶**

A time series is a set of observations ordered in time usually collected at regular intervals.

<br/>

**How is time series different from the usual regression problem? 👶**

The principle behind causal forecasting is that the value that has to be predicted is dependant on the input features (causal factors). In time series forecasting, the to be predicted value is expected to follow a certain pattern over time.

<br/>

**Which models do you know for solving time series problems? ‍⭐️**

* Simple Exponential Smoothing: approximate the time series with an exponentional function
* Trend-Corrected Exponential Smoothing (Holt‘s Method): exponential smoothing that also models the trend
* Trend- and Seasonality-Corrected Exponential Smoothing (Holt-Winter‘s Method): exponential smoothing that also models trend and seasonality
* Time Series Decomposition: decomposed a time series into the four components trend, seasonal variation, cycling varation and irregular component
* Autoregressive models: similar to multiple linear regression, except that the dependent variable y_t depends on its own previous values rather than other independent variables.
* Deep learning approaches (RNN, LSTM, etc.)

<br/>

**If there’s a trend in our series, how we can remove it? And why would we want to do it? ‍⭐️**

We can explicitly model the trend (and/or seasonality) with approaches such as Holt's Method or Holt-Winter's Method. We want to explicitly model the trend to reach the stationarity property for the data. Many time series approaches require stationarity. Without stationarity,the interpretation of the results of these analyses is problematic [Manuca, Radu & Savit, Robert. (1996). Stationarity and nonstationarity in time series analysis. Physica D: Nonlinear Phenomena. 99. 134-161. 10.1016/S0167-2789(96)00139-X. ].

<br/>

**You have a series with only one variable “y” measured at time t. How do predict “y” at time t+1? Which approaches would you use? ‍⭐️**

We want to look at the correlation between different observations of y. This measure of correlation is called autocorrelation. Autoregressive models are multiple regression models where the time-lag series of the original time series are treated like multiple independent variables.

<br/>

**You have a series with a variable “y” and a set of features. How do you predict “y” at t+1? Which approaches would you use? ‍⭐️**

Given the assumption that the set of features gives a meaningful causation to y, a causal forecasting approach such as linear regression or multiple nonlinear regression might be useful. In case there is a lot of data and the explanability of the results is not a high priority, we can also consider deep learning approaches.

<br/>

**What are the problems with using trees for solving time series problems? ‍⭐️**

Random Forest models are not able to extrapolate time series data and understand increasing/decreasing trends. It will provide us with average data points if the validation data has values greater than the training data points.

<br/>
