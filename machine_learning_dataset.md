====================
MACHINE LEARNING
====================

1. Introduction to Machine Learning
-----------------------------------
**Definition:**
Machine Learning (ML) is the field of study that gives computers the ability to learn patterns and make decisions from data without being explicitly programmed for every rule. ML systems can infer mappings from inputs (features) to outputs (labels) or discover structure in unlabeled data.


**High-level Types of Learning:**
- **Supervised Learning:** Learning from labeled examples. Tasks include classification and regression.
- **Unsupervised Learning:** Discovering structure in unlabeled data such as clustering, dimensionality reduction.
- **Reinforcement Learning (RL):** An agent interacts with an environment, learning through trial-and-error to maximize cumulative reward.

**ML Pipeline (Detailed):**
1. **Problem Definition:**
   - Specify the task (classification, regression, clustering, anomaly detection).
        - Classification : 
            Definition : Classification is a supervised learning task where the goal is to predict a categorical label (discrete outcome) for given input data.

            Example:

                - Email → Spam or Not Spam
                - Tumor → Malignant or Benign
                - Image → Cat, Dog, or Horse

            Working Principle

                The model is trained using labeled data (i.e., inputs with known class labels).
                It learns the relationship between features (X) and classes (Y).
                For new input data, it predicts which class it belongs to.

            Mathematically:
            f(X)→Yclass
            is from a finite set of labels, e.g. {0, 1, 2, ...}.

        - Regression : 
            Definition : Regression is a supervised learning task used to predict a continuous numeric value based on input features.

            Example:

                - Predicting house prices
                - Estimating temperature
                - Forecasting sales or stock prices

            Working Principle : The model learns a function that maps input variables 

                f(X)→Y_continuous

        - Clustering : 
            Definition : Clustering is an unsupervised learning task that groups similar data points into clusters such that:
                - Points within the same cluster are similar to each other.
                - Points in different clusters are dissimilar.

            Example:

                - Grouping customers by purchasing behavior
                - Segmenting images by color or texture
                - Grouping documents by topic

            Working Principle : There are no labels — the model only looks at feature similarity (e.g., distance or density). It tries to discover hidden patterns or natural groupings in the data.

        - Anamoly Detection
            Definition : Anomaly Detection identifies unusual data points that deviate significantly from the normal pattern of the dataset.

            Example:

                - Detecting credit card fraud
                - Identifying faulty sensors
                - Spotting network intrusions
                - Medical anomaly diagnosis

            Working Principle : The model learns the normal pattern of the data. Points that differ greatly from the learned pattern are marked as anomalies or outliers.

            Mathematically:
                If ∣xi−μ∣>kσ, then xi is an anomaly.
            
    - Determine evaluation metrics (accuracy, F1-score, RMSE, silhouette score).

2. **Data Collection and Labeling:**
   - Gather raw data from sensors, logs, or APIs.
   - Labeling may involve manual annotation or automatic heuristics.
   - Mathematical note: Dataset D = {(x_i, y_i)}, i = 1,...,n

3. **Data Cleaning and Preprocessing:**
   - Handle missing values: mean/mode/median imputation, interpolation.
   - Normalize/standardize features: z = (x - μ)/σ
   - Encode categorical features: One-hot encoding, label encoding
   - Detect outliers: Z-score, IQR method

**Python Examples:**
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# Handling missing values
df.fillna(df.mean(), inplace=True)
# Standardization
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['feature1','feature2']])
# One-hot encoding
encoder = OneHotEncoder()
categorical_encoded = encoder.fit_transform(df[['category']])
```

4. **Feature Engineering / Representation:**
   - Create new features from raw data: e.g., ratio, log-transform, polynomial features
   - Feature selection: remove irrelevant/redundant features (VarianceThreshold, Recursive Feature Elimination)
   - Dimensionality reduction: PCA, LDA, t-SNE for visualization or reducing noise

**Mathematical Note:**
- PCA: project data X onto k principal components by eigen decomposition of covariance matrix Σ = X^T X / (n-1)
- Feature importance: based on model coefficients (linear models) or tree-based feature importance scores

5. **Model Selection and Training:**
   - Select algorithm based on problem and data size: Linear Regression, Logistic Regression, Decision Trees, SVM, KNN, Naive Bayes, Neural Networks
   - Split data: Train/Validation/Test (commonly 70/15/15%)
   - Hyperparameter tuning: grid search, random search, Bayesian optimization

**Mathematical Concept:**
- Minimize loss function J(θ) = (1/n) Σ L(y_i, ŷ_i) over parameters θ
- Update weights using Gradient Descent:
    θ := θ - α ∇J(θ)

**Python Implementation (example with Grid Search):**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
param_grid = {'n_estimators':[50,100,200], 'max_depth':[5,10,20]}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
```

6. **Evaluation:**
   - Use validation set or cross-validation to prevent overfitting
   - Metrics depend on task:
       - Regression: MSE, RMSE, MAE, R²
       - Classification: Accuracy, Precision, Recall, F1-score, ROC-AUC

**Cross-validation:**
- K-Fold: Split data into k folds, train on k-1 folds, validate on 1 fold, repeat k times
- Leave-One-Out: Special case of k-fold with k=n

**Python Example:**
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
mean_score = scores.mean()
```

7. **Deployment and Monitoring:**
   - Export trained model (Pickle, joblib, ONNX)
   - Serve via REST API, batch inference, or embedded system
   - Monitor model performance over time to detect data drift
   - Retrain model periodically if performance drops

8. **Common Terminology & Significance:**
- **Feature:** Input variable; quality and relevance significantly impact model performance
- **Label/Target:** Ground truth for supervised learning
- **Overfitting:** Model memorizes training data; reduces generalization
- **Underfitting:** Model is too simple; fails to capture patterns
- **Hyperparameter:** Controls learning process; tuned to optimize performance
- **Significance of Technologies:**
   - Scikit-learn: versatile ML library with most classic algorithms
   - TensorFlow/PyTorch: Deep learning frameworks for neural networks
   - XGBoost/LightGBM/CatBoost: Gradient boosting libraries for structured data
   - Pandas/Numpy: Data manipulation and computation


![Detailed ML pipeline showing all stages from data collection to deployment](ml_pipeline.jpg)

====================
SUPERVISED LEARNING
====================

1. Overview of Supervised Learning
----------------------------------
Supervised Learning is a Machine Learning paradigm where the model is trained on labeled data. The goal is to learn a mapping function f(X) = Y that predicts the output Y from input features X. It encompasses two main categories:

- Regression: Predicting continuous values.
- Classification: Predicting discrete labels.

Workflow includes:
1. Data collection & preprocessing
2. Train-validation-test split
3. Model selection & training
4. Evaluation & tuning
5. Deployment

Key concepts:
- Loss function: Measures prediction error.
- Regularization: Prevents overfitting.
- Cross-validation: Evaluates model performance.
- Feature scaling/normalization: Necessary for distance-based models.

![Supervised learning workflow diagram showing data -> model -> predictions](supervised_learning.png)

2. Linear Regression
-------------------
**Introduction**
Linear Regression is one of the most fundamental algorithms in Machine Learning, used to model the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation.
It assumes the relationship between the variables is approximately linear.

Equation:
     $$ \hat{y} = \beta_0 + \sum_{j=1}^{d} \beta_j x_j $$

- y: dependent variable (output)
- xi : independent variables (inputs/features)
- β0: Intercept
- βi: Coefficients
- ε: Error term

**Objective**
The goal is to find parameters β that minimize the sum of squared errors (SSE) between predicted and actual outputs.
Minimize Residual Sum of Squares (RSS):
    $$J(\beta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Matrix Formulation**
X = [[1 x11 x12 ... x1p], ..., [1 xn1 xn2 ... xnp]], y = [y1,...,yn], β = [β0,...,βp]
    ŷ = Xβ

Normal Equation:
    $$\beta = (X^T X)^{-1} X^T y$$

**Gradient Descent Update**
    Gradient Descent is an optimization algorithm used to minimize the cost (or loss) function in machine learning models.

    In simple words:

    Gradient Descent helps a model learn by adjusting its parameters (weights and biases) in the direction that reduces the error.

    It’s like trying to go down a hill to reach the lowest point — the minimum of the cost function.

    Intuitive Example

    Imagine you’re standing on a mountain in the dark.
    Your goal is to reach the lowest valley (minimum), but you can’t see far — only feel the slope beneath your feet.

    You take small steps downhill (in the direction of the steepest descent) until you reach the bottom.
    That’s gradient descent.

    Suppose we have:

    A cost function J(θ) that we want to minimize
$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} L\big(\hat{y}^{(i)}, y^{(i)}\big)
$$


    where:

    ∂J(θ)/∂θ_j → gradient (slope of the cost function w.r.t parameter )
    α → learning rate (controls the step size)

    Working Principle

        1. Initialize parameters (randomly or with zeros).
        2. Compute predictions using current parameters.
        3. Calculate cost 
        4. J(θ) (how far predictions are from real values).
        5. Compute gradients (partial derivatives of the cost function).
        6. Update parameters in the opposite direction of the gradient.
        7. Repeat steps 2–5 until convergence (cost stops changing significantly).

    Role of the Learning Rate (α)

        Too Small (α → 0) =	Converges slowly (tiny steps downhill)
        Too Large (α → big) =	May overshoot the minimum or diverge
    Just Right	Smooth and fast convergence

    * Types of Gradient Descent *

        1️⃣ Batch Gradient Descent
            Definition : In Batch Gradient Descent, the algorithm uses the entire training dataset to compute the gradient of the cost function before performing a single update to the parameters. This means one update per epoch (i.e., after one full pass through all data).

            Mathematical Formulation

            For a dataset with 
            m training examples:

$$ \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \frac{\partial J^{(i)}(\theta)}{\partial \theta_j} $$

            Here:

            J(i)(θ) = loss for the ith sample

            ∂J(i)(θ)/∂θ_j = gradient contribution from that sample

            Intuition

                Imagine computing the exact slope of the mountain using all available data before taking one careful step. This gives an accurate but slow movement toward the minimum.

            Advantages

            ✅ Converges smoothly and steadily (less noise)
            ✅ Produces accurate gradient estimation
            ✅ Suitable for small or medium-sized datasets

            Disadvantages

            ❌ Very slow for large datasets (you need to process the entire data before one update)
            ❌ Memory-intensive — must load all data into memory
            ❌ Can get stuck in local minima for non-convex functions

            Use Case

                - When the dataset is small enough to fit into memory.
                - Used in classical ML models (like Linear Regression) where data size is manageable.

        2️⃣ Stochastic Gradient Descent (SGD)
            Definition : In Stochastic Gradient Descent, the parameters are updated for each individual training example. That is, the model takes one data point at a time, computes the gradient, and updates immediately.

            Mathematical Formulation
$$\theta_j := \theta_j - \alpha \frac{\partial J^{(i)}(\theta)}{\partial \theta_j}
$$
            (for each training sample i)

            Intuition

                Instead of computing the exact slope, you take a rough estimate using only one point. This makes learning faster but noisier.  The noise sometimes helps the algorithm escape local minima and find a better global solution.

            Advantages

            ✅ Very fast updates (especially on large datasets)
            ✅ Can escape local minima due to random fluctuations
            ✅ Ideal for online learning (continuous incoming data)

            Disadvantages

            ❌ Highly noisy updates — loss function fluctuates
            ❌ May overshoot the minimum
            ❌ Convergence is less stable

            Use Case

                - When the dataset is very large or streaming (e.g., online ad click prediction, IoT data).
                - Common in deep learning when combined with optimizers like Adam or RMSProp.

        3️⃣ Mini-Batch Gradient Descent
            Definition : Mini-Batch Gradient Descent is a compromise between Batch and SGD. It splits the dataset into small batches (e.g., 32, 64, 128 samples per batch) and performs an update for each batch. So instead of using 1 sample or all samples, it uses a subset each time.

            Mathematical Formulation
$$\theta_j := \theta_j - \alpha \frac{1}{b} \sum_{i=1}^{b} \frac{\partial J^{(i)}(\theta)}{\partial \theta_j}$$

            where 
            b = batch size (e.g., 32)

            Intuition

                It gives a balance between accuracy (like batch) and speed (like SGD).
                Reduces the noise of SGD but maintains fast convergence.

            Advantages

            ✅ Faster convergence than batch gradient descent
            ✅ More stable than pure SGD (less noisy)
            ✅ Exploits parallel computation (GPUs) efficiently
            ✅ Works best for deep learning models

            Disadvantages

            ❌ Requires tuning of batch size
            ❌ May still have slight oscillations in cost
            ❌ Needs careful choice of learning rate

            Use Case

                - Default choice in deep learning frameworks (TensorFlow, PyTorch, Keras, etc.)
                - Batch size typically = 32, 64, or 128

        4️⃣ Momentum-based Gradient Descent
            Intuition:

                Regular Gradient Descent can oscillate a lot — especially in regions with steep and flat gradients (ravines).
                Momentum adds a concept from physics:

                “Keep moving in the same direction unless a strong opposite force acts.”

                It helps smooth the path toward the minimum and accelerates convergence.

                Mathematical Formulation:

                Let:
                    vt = velocity (running average of gradients)
                    β = momentum coefficient (usually 0.9)
                    α = learning rate

                Then:
$$v_t = \beta v_{t-1} + (1 - \beta) \nabla_\theta J(\theta)$$

$$\theta := \theta - \alpha v_t$$

                Key Idea:

                vt accumulates gradient information over time.

                This smooths updates and avoids sharp changes.

                Momentum allows faster movement along consistent gradient directions.

                Pros:

                    ✅ Faster convergence
                    ✅ Less oscillation
                    ✅ More stable path toward minimum

        5️⃣ Nesterov Accelerated Gradient (NAG)
            Intuition:

            Nesterov improves on momentum by taking a lookahead step.
            Before computing the gradient, it estimates the next position based on current velocity — then calculates the gradient there.

            This gives a more accurate correction of the direction.

            Mathematical Formulation:
$$v_t = \beta v_{t-1} + (1 - \beta) \nabla_\theta J(\theta - \alpha \beta v_{t-1})$$

$$\theta := \theta - \alpha v_t$$

            Intuition in Words:

                Momentum looks backward (based on the previous step).
                NAG “looks ahead” to where the momentum is taking us, and corrects the direction before actually reaching that point.

            Pros:

                ✅ Anticipates future position
                ✅ More responsive to curvature
                ✅ Usually converges faster than plain Momentum

        6️⃣ Adagrad (Adaptive Gradient Algorithm)
            Intuition:

            Adagrad adapts the learning rate for each parameter individually based on how frequently it has been updated.

            Parameters with large/frequent gradients → smaller learning rates

            Parameters with small/infrequent gradients → larger learning rates

            Great for sparse data (like NLP or recommender systems).

            Mathematical Formulation:
$$
G_{jj} = \sum_{t=1}^{T} \left( \frac{\partial J(\theta_t)}{\partial \theta_j} \right)^2
$$
$$
\theta_j := \theta_j - \frac{\alpha}{\sqrt{G_{jj}} + \epsilon} \frac{\partial J(\theta)}{\partial \theta_j}
$$

            Key Idea:

                Gjj stores the sum of squared gradients for parameter θj.
                The denominator increases as the parameter is updated often, reducing its learning rate.

            Pros:

                ✅ Automatic learning rate adjustment
                ✅ Works well for sparse features

            Cons:

                ❌ Learning rate keeps decreasing → may stop learning eventually
        
        7️⃣ RMSProp (Root Mean Square Propagation)
        Intuition:

        RMSProp fixes Adagrad’s problem by using an exponentially decaying average of squared gradients instead of accumulating all past gradients.

        This keeps the learning rate adaptive but stable.

        Mathematical Formulation:
                
$$
E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) g_t^2
$$
$$
\theta := \theta - \frac{\alpha}{\sqrt{E[g^2]_t} + \epsilon} g_t
$$

        Key Idea:

            Keeps track of recent squared gradients using exponential decay.
            Divides learning rate by the root of this moving average → stabilizes updates.

        Pros:

            ✅ Works well in non-stationary settings
            ✅ Prevents vanishing learning rate
            ✅ Very popular for deep networks

        8️⃣ Adam (Adaptive Moment Estimation)
        Intuition:

Adam = Momentum + RMSProp
It combines:

Momentum’s running average of gradients

RMSProp’s running average of squared gradients

Result: Adaptive + stable + fast convergence

Mathematical Formulation:

1️⃣ First moment estimate (mean):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

2️⃣ Second moment estimate (variance):
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

3️⃣ Bias correction:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

4️⃣ Parameter update:
$$\theta := \theta - \frac{\alpha \, \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Pros:

    ✅ Adaptive learning rate
    ✅ Fast and stable convergence
    ✅ Works very well for deep learning models

Cons:

    ❌ Sometimes overshoots local minima
    ❌ Slightly more computation and memory cost

**Assumptions**
1. Linearity
2. Independence of errors
3. Homoscedasticity
4. Normality of residuals
5. No multicollinearity

**Evaluation Metrics**
- MSE : $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
- RMSE : $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
- MAE : $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
- R^2 : $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

**Example (manual calculation)**
Hours studied vs marks example. β1=0.7, β0=2.1, ŷ = 2.1 + 0.7*x

**Visualization**
![Scatter plot with regression line showing residuals](lr.png)

**Python Implementation**
```python
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,5,4,5])
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
print(model.intercept_, model.coef_)
```

**Regularization**
- Ridge (L2), Lasso (L1), ElasticNet

**Advantages**
- Simple, interpretable, computationally efficient
**Disadvantages**
- Assumes linearity, sensitive to outliers

**Applications**
- House prices, sales forecasting, salary prediction

3. Logistic Regression
----------------------
**Introduction**
Logistic Regression is a classification algorithm used to predict binary outcomes (e.g., Yes/No, 0/1, Spam/Not Spam). Despite its name, it is a linear model — it predicts the probability of an outcome using a logistic (sigmoid) function.

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

where 
$$\quad z = \beta_0 + \sum_{j=1}^{d} \beta_j x_j$$

Output of sigmoid:

   - Close to 1 → event likely to occur
   - Close to 0 → event unlikely to occur

Example:
If output = 0.85 → 85% chance that class = 1

Equation:
    P(y=1|X) = 1 / (1 + exp(-(β0 + β1*x1 + ... + βp*xp)))

Decision Rule:
    ŷ = 1 if P > 0.5 else 0

Log-Odds (Logit) Form

Taking the log of the odds gives a linear relationship:

log⁡(P(y=1)/(1−P(y=1))) = β0 + β1x1 + β2x2 + ⋯+ βnxn

This is why logistic regression is still considered linear in the parameters (the log-odds are linear).

**Loss Function**
Since logistic regression uses probabilities, we use the log loss (cross-entropy loss) instead of mean squared error;
Cross-entropy:
    $$J(\beta) = -\frac{1}{n} \sum_{i=1}^{n} \left[y_i \log(\hat{y}_i) + (1-y_i) \log(1 - \hat{y}_i)\right]$$

**Gradient Descent Update**
    $$\beta_j := \beta_j - \alpha \frac{\partial L}{\partial \beta_j}$$

**Pseudo-code**
    Initialize β randomly
    Repeat:
        p_i = sigmoid(Xβ)
        error = p_i - y
        β = β - α * (1/n) * X^T * error

**Assumptions**
- Linearity of logit, independence, little multicollinearity, large sample size preferred

**Evaluation Metrics**
- Accuracy, Precision, Recall, F1-score, ROC-AUC
![Sigmoid curve showing probability vs input z](logistic_prob_curve.jpeg)
![Decision boundary separating two classes](logistic_decision.png)

**Python Implementation**
```python
from sklearn.linear_model import LogisticRegression
X = np.array([[0],[1],[2],[3],[4],[5]])
y = np.array([0,0,0,1,1,1])
model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)
```

**Advantages**
- Outputs probabilities, interpretable, works for linearly separable data
**Disadvantages**
- Assumes linear decision boundary, underperforms with complex patterns

**Applications**
- Spam detection, disease diagnosis, customer churn prediction

![Confusion matrix example diagram](confusion_matrix.png)

4. Decision Trees
------------------
**Introduction**
A Decision Tree is a supervised learning algorithm that can be used for classification (predicting categories) and regression (predicting continuous values).

It splits a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed.
The final result is a tree with decision nodes and leaf nodes.

    Decision Node: Represents a test on a feature (e.g., “Age < 30?”)
    Branch: Outcome of the test (Yes/No or True/False)
    Leaf Node: Final output or decision (e.g., “Buys Computer = Yes”)

This process is similar to how humans make decisions step-by-step.

**Algorithm**
The algorithm works recursively using a top-down approach called recursive partitioning.

Step 1: Select the Best Attribute

    The tree starts with the entire dataset.
    It selects the feature that best splits the data (i.e., makes subsets more “pure”).

Step 2: Create Subsets

    Data is split into subsets according to feature values.

Step 3: Repeat

    For each subset, repeat the process (find best feature, split again).

Step 4: Stop

    Stop when: 
        All samples belong to one class.
        No features remain.
        Maximum tree depth is reached.

**Splitting Criteria**
To decide which feature gives the “best split,” we use impurity measures.

(a) Gini Impurity : Used in CART (Classification and Regression Trees).

$$G = 1 - \sum_{k=1}^{K} p_k^2$$

where 
    pi = proportion of samples of class i in the node.

Gini = 0 → perfectly pure node
Gini = 0.5 → completely mixed node (two classes equally likely)

(b) Entropy and Information Gain : Used in ID3, C4.5 algorithms.

$$ Entropy = -\sum_{k=1}^{K} p_k \log_2 p_k$$

Information Gain (IG):

IG=Entropyparent−∑kNkNparentEntropyk

Choose the feature with highest Information Gain — it reduces disorder the most.

(c) For Regression Trees

Use:

Mean Squared Error (MSE)

Mean Absolute Error (MAE)
to find the best split.

Gini Impurity:
$$G(D) = 1 - \sum_{k=1}^{K} p_k^2$$

Entropy:
$$ Entropy(D) = -\sum_{k=1}^{K} p_k \log_2 p_k$$

**Pseudo-code**
```
If stopping criteria met:
    make leaf node
Else:
    for each feature:
        compute impurity reduction
    select best feature & threshold
    split data into left & right
    recurse on child nodes
```

**Advantages**
- Interpretable, handles non-linear relationships, works with mixed data types
**Disadvantages**
- Prone to overfitting, high variance

**Python Implementation**
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

![Example decision tree diagram](decision_tree.png)

5. Random Forest
----------------
**Introduction**
A Random Forest is an ensemble learning algorithm that combines the predictions of multiple Decision Trees to improve accuracy and reduce overfitting.

It is used for both:

    Classification (predicting categories)
    Regression (predicting continuous values)

The idea is simple but powerful:

    “Train many weak Decision Trees and combine their outputs to form a strong predictor.”

Intuition

    A single Decision Tree can overfit the training data (high variance).
    By averaging multiple trees trained on different subsets of data and features, Random Forest reduces overfitting and increases robustness.

    It’s like taking opinions from many people rather than relying on one person’s judgment.

**Algorithm**
1. Bootstrap Sampling

    From the training data of size N, create B random samples with replacement (called bootstrap samples).
    Each sample is used to train one Decision Tree.

    This process is called Bagging (Bootstrap Aggregating).

2. Tree Construction

    Each tree is trained on a different bootstrap sample.
    At each split in the tree, instead of considering all features, a random subset of features (of size m) is chosen.
    The best split is found only among those m features.

3. Prediction (Ensemble Step)

    For classification, each tree votes for a class → majority vote is final.
    For regression, average the outputs of all trees.

Mathematical Representation
    Classification:

    Each tree Tb produces a prediction yb_hat(x).
    The final output is the mode (most frequent class):
$$\hat{y} = \text{mode}\left( \hat{y}_1(x), \hat{y}_2(x), \ldots, \hat{y}_B(x) \right)$$

    Regression:

    Each tree gives a continuous output, and the final prediction is the average:
$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} \hat{y}_b(x)$$

**Key Parameters**

n_estimators = Number of trees in the forest
max_features = Number of features to consider at each split
max_depth = Maximum depth of each tree
min_samples_split = Minimum samples required to split a node
bootstrap = Whether to use bootstrapping for samples
random_state = Controls randomness for reproducibility

**Out-of-Bag (OOB) Error**

Since each tree is trained on a bootstrap sample (~63% of total data), about 37% of the data is not used for that tree.
This unused data is called Out-of-Bag data.

It serves as a built-in validation set.

OOB error gives an unbiased estimate of test accuracy without using cross-validation.

$$\text{OOB Error} = \frac{1}{N} \sum_{i=1}^{N} I(y_i \neq \hat{y}_{\text{OOB}}^{(i)})$$

where,
I(⋅) is the indicator function.

**Advantages**
    ✅ Reduces overfitting (because of averaging)
    ✅ Works for both classification & regression
    ✅ Handles missing data well
    ✅ Can estimate feature importance
    ✅ Robust to noise and outliers
    ✅ Good accuracy even without parameter tuning

**Disadvantages**
    ❌ Slower to train and predict (many trees)
    ❌ Harder to interpret than a single Decision Tree
    ❌ Memory intensive (stores multiple models)
    ❌ May not perform well on very high-dimensional sparse data

**Python Implementation**
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

![Random forest ensemble illustration](random_tree.png)

6. Support Vector Machine (SVM)
--------------------------------
**Introduction**
A Support Vector Machine (SVM) is a supervised learning algorithm used for both classification and regression problems — but it is mostly used for classification.

SVM works by finding the optimal hyperplane that best separates data points of different classes with the maximum margin.

Intuition

    The idea is to: “Find a boundary (hyperplane) that not only separates classes but does so with the largest possible margin.”

    The margin is the distance between the hyperplane and the closest data points (called support vectors).
    These support vectors are the critical elements that define the decision boundary.

**Geometrical Concept**

For a 2D case:

The hyperplane becomes a line.

The goal is to find a line that separates two classes and maximizes the distance from the nearest points of each class.

For higher dimensions:

The hyperplane becomes a plane or n-dimensional surface.


**Mathematical Formulation**
    $$\min_{w,b} \frac{1}{2} ||w||^2 \quad \text{s.t.} \quad y_i (w^T x_i + b) \ge 1, ; i=1,\dots,n$$

    For correctly classified points:

The following constraints should hold:
$$y_i (w^T x_i + b) \geq 1 \quad \forall i$$

Margin Maximization

The margin (M) is given by:
M = 2/||w||

So, maximizing margin M is equivalent to minimizing ∥w∥.

Hence, the optimization problem becomes:
$$\min_{w, b} \frac{1}{2} \|w\|^2$$

subject to:
$$\text{subject to: } y_i (w^T x_i + b) \geq 1, \quad \forall i$$

**Soft Margin SVM (with Slack Variables)**

Real-world data is not always perfectly separable.
So, slack variables 
ξi≥0 are introduced to allow some misclassifications.

$$y_i (w^T x_i + b) \geq 1 - \xi_i$$

The modified optimization function becomes:
$$\min_{w, b, \xi} \left( \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \xi_i \right)$$

where,
C is a regularization parameter that controls the trade-off between margin size and classification error.

Dual Form and Lagrange Multipliers

Using Lagrange multipliers (αi), the dual form of the optimization problem becomes:
$$\max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j (x_i^T x_j)$$

subject to : $$\text{subject to: } \sum_{i=1}^{n} \alpha_i y_i = 0, \quad \alpha_i \geq 0$$

**Kernels**
If data is not linearly separable, SVM can project it into a higher-dimensional space using a kernel function.

This allows the algorithm to find a linear separation in that high-dimensional feature space.

Common kernel functions:

1. Linear Kernel
$$K(x_i, x_j) = x_i^T x_j$$

2. Polynomial Kernel
$$K(x_i, x_j) = (x_i^T x_j + c)^d$$

3. RBF Kernel
$$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$$

4. Sigmoid Kernel
$$K(x_i, x_j) = \tanh(\alpha x_i^T x_j + c)$$

**Decision Function**

Once training is complete, the prediction for a new data point x is given by:
$$f(x) = \text{sign}\left( \sum_{i=1}^{n} \alpha_i y_i K(x_i, x) + b \right)$$

**Pseudo-code**
```
Compute optimal hyperplane using quadratic programming
If non-linear: apply kernel transformation
Predict class based on side of hyperplane
```

**Advantages**
    ✅ Works well for high-dimensional spaces
    ✅ Effective even when number of features > samples
    ✅ Uses only support vectors → memory efficient
    ✅ Can handle non-linear data via kernels
    ✅ Robust to overfitting (especially with proper regularization)
**Disadvantages**
    ❌ Computationally expensive for large datasets
    ❌ Difficult to choose the right kernel and parameters
    ❌ No direct probabilistic interpretation (unlike logistic regression)
    ❌ Slower training time for very large data

**Python Implementation**
```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

![SVM margin and support vectors diagram](svm.png)

7. K-Nearest Neighbors (KNN)
-----------------------------
**Introduction**
K-Nearest Neighbors (KNN) is a non-parametric, instance-based, and lazy learning algorithm used for both classification and regression.

It does not learn an explicit model.
Instead, it memorizes the training data and makes predictions based on the K closest training examples in the feature space.

Intuition

    The core idea is:

    “If it walks like a duck and quacks like a duck — it’s probably a duck!”

    That is, data points that are close to each other in feature space are likely to belong to the same class (for classification) or have similar output values (for regression).

**Algorithm**
1. Choose the number of neighbors K.
2. For a new data point x:
    Calculate its distance from all training samples.

3. Identify the K nearest neighbors based on the smallest distances.
4. Classification:
    Take a majority vote among the neighbors’ labels.
5. Regression:

    Take the average (or weighted average) of the neighbors’ values.

**Distance Metrics**
To measure similarity, KNN uses distance functions between points.
Commonly used ones are:

(a) Euclidean Distance
$$d(x, y) = \sqrt{ \sum_{i=1}^{n} (x_i - y_i)^2 }$$

(b) Manhattan Distance
$$d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$$

(c) Minkowski Distance (Generalized Form)
$$d(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{1/p}$$

**Prediction Rules**
(a) Classification

For a new sample x, let its K nearest neighbors be 
NK(x).
The predicted class $$\hat{y}$$ is the most frequent class among those neighbors:
$$\hat{y} = \text{mode} \left( \{ y_i : x_i \in N_K(x) \} \right)$$

(b) Regression

The prediction is the average of the neighbors’ values:
$$\hat{y} = \frac{1}{K} \sum_{x_i \in N_K(x)} y_i$$

(c) Weighted KNN

Closer neighbors can be given higher weights (inverse of distance):
$$\hat{y} = \frac{ \sum_{x_i \in N_K(x)} \frac{y_i}{d(x, x_i)} }{ \sum_{x_i \in N_K(x)} \frac{1}{d(x, x_i)} }$$

**Choosing the Value of K**

    Small K → High variance, low bias → overfitting
    Large K → High bias, low variance → underfitting

Typical choice: K is an odd number (to avoid ties in classification).

**Pseudo-code**
```
Compute distance d_i = distance(x0, x_i)
Sort distances, pick k nearest
For classification: majority label
For regression: mean value
```
**Advantages**
    ✅ Simple and intuitive to understand
    ✅ No training phase (lazy learner)
    ✅ Works well for smaller datasets
    ✅ Naturally handles multi-class problems
    ✅ Adapts well to non-linear decision boundaries
**Disadvantages**
    ❌ Computationally expensive during prediction (needs distance from all points)
    ❌ Sensitive to the scale of data (feature normalization required)
    ❌ Performance degrades in high dimensions (curse of dimensionality)
    ❌ Requires storage of the entire dataset
    ❌ Struggles with imbalanced data

**Python Implementation**
```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

![KNN decision boundaries illustration](knn.png)

8. Naive Bayes
---------------
**Introduction**
Naïve Bayes is a probabilistic supervised learning algorithm based on Bayes’ Theorem, primarily used for classification tasks.

It assumes that:

“All features are conditionally independent of each other given the class label.”

Despite this “naïve” independence assumption, it performs remarkably well in many real-world problems — especially in text classification and spam detection.

Intuition

Naïve Bayes calculates the posterior probability of each class given an input, and assigns the class with the highest posterior probability.
$$P(C_k | X) = \frac{P(X | C_k) \cdot P(C_k)}{P(X)}$$

**Formula**
Bayes’ Theorem

Bayes’ theorem relates conditional probabilities:
$$P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}$$

Where:
P(A∣B): Posterior probability (probability of A given B)
P(B∣A): Likelihood (probability of B given A)
P(A): Prior probability of A
P(B): Evidence or normalization constant

Applying Bayes’ Theorem to Classification

Let 
X=(x1,x2,...,xn)be a feature vector.
We compute the probability of each class Ck given the features:
$$P(C_k | X) = \frac{P(x_1, x_2, ..., x_n | C_k) \cdot P(C_k)}{P(x_1, x_2, ..., x_n)}$$

Since 
P(x1,x2,...,xn) is constant for all classes, it can be ignored for comparison.
So we predict the class as:
$$\hat{C} = \arg\max_{C_k} \left[ P(C_k) \cdot P(x_1, x_2, ..., x_n | C_k) \right]$$

**Naïve Conditional Independence Assumption**

Naïve Bayes assumes that all features xi are independent given the class label Ck:
$$P(x_1, x_2, ..., x_n | C_k) = \prod_{i=1}^{n} P(x_i | C_k)$$

Thus, the prediction rule simplifies to:
$$\hat{C} = \arg\max_{C_k} \left[ P(C_k) \prod_{i=1}^{n} P(x_i | C_k) \right]$$

**Types**
(a) Gaussian Naïve Bayes

Used when features are continuous and assumed to follow a normal (Gaussian) distribution.

The likelihood is given by:
$$P(x_i | C_k) = \frac{1}{\sqrt{2\pi\sigma_{k,i}^2}} \exp\left( -\frac{(x_i - \mu_{k,i})^2}{2\sigma_{k,i}^2} \right)$$

Where:
$$\mu_{k,i}$$: Mean of feature $${x_i}$$for class Ck.
 $$\sigma_{k,i}$$: Standard deviation of feature xi for class Ck​

(b) Multinomial Naïve Bayes

Used for discrete features, e.g., word counts in text classification.
$$P(X | C_k) = \frac{(\sum_i x_i)!}{\prod_i x_i!} \prod_i P(x_i | C_k)^{x_i}$$

(c) Bernoulli Naïve Bayes

Used for binary/boolean features (e.g., word presence or absence).
$$P(X | C_k) = \prod_{i=1}^{n} P(x_i | C_k)^{x_i} \cdot (1 - P(x_i | C_k))^{(1 - x_i)}$$

Laplace (Additive) Smoothing

To handle zero probabilities (when a feature never occurs in a class), we add a small constant α:
$$P(x_i | C_k) = \frac{N_{x_i, C_k} + \alpha}{N_{C_k} + \alpha \cdot n}$$

Where:
Nxi,Ck: Count of feature $${x_i}$$ in class Ck
N_Ck: Total feature occurrences in class Ck
n: Number of unique features
α: Smoothing parameter (commonly 1)

**Advantages**
    ✅ Fast and simple to implement
    ✅ Works well with high-dimensional data (e.g., text)
    ✅ Requires small training data
    ✅ Handles multi-class problems
    ✅ Robust to irrelevant features
**Disadvantages**
    ❌ Assumes feature independence (often unrealistic)
    ❌ Poor performance when features are correlated
    ❌ Continuous features require proper distribution assumptions
    ❌ Output probabilities may not be well-calibrated

**Python Implementation**
```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

![Naive Bayes probability distributions diagram](nb.jpg)

9. Gradient Boosting (AdaBoost & XGBoost)
-----------------------------------------
**Introduction**
Gradient Boosting is an ensemble technique that builds a strong model by combining multiple weak learners (usually decision trees) sequentially.

Each new model tries to correct the errors (residuals) made by the previous models.

It is a boosting algorithm, meaning it:

    Builds models one after another
    Each model focuses on errors of the previous one

Intuition

    "Boosting = Learning from Mistakes."

    Unlike bagging (which trains models independently), boosting adds models sequentially such that each new learner focuses on the examples that previous learners got wrong.

**Algorithm**
Given training data , the goal is to minimize a loss function.
1. Initialize model with simple,constant predictor
$$F_0(x) = \arg\min_c \sum_{i=1}^{n} L(y_i, c)$$

2. For each iteration m=1,2,.....,M:
    - Compute pseudo-residuals (negative gradient of loss): $$r_{im} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x)=F_{m-1}(x)}$$

    - Fit a weak learner h_m(x) to predict r_im.
    - Compute multiplier (step size):
    $$\gamma_m = \arg\min_\gamma \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$$

    - Update the model:
    $$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$

3. Final model after M iterations:
$$F_M(x) = \sum_{m=0}^{M} \gamma_m h_m(x)$$

**Types**
1) AdaBoost (Adaptive Boosting)
AdaBoost adjusts the weights of training samples to focus on those that were misclassified by previous weak learners.

Intuition of AdaBoost

    If a sample is classified correctly, its weight is reduced.
    If it is misclassified, its weight is increased, forcing the next learner to focus on it.

Algorithm:
1. Assign equal weights to all training samples:
$$w_i = \frac{1}{N}$$

2. Train a weak learner h_m(x) using weights $$w_i$$. 
3. Compute the weighted error:
$$\varepsilon_m = \frac{\sum_{i=1}^{N} w_i \cdot I(y_i \neq h_m(x_i))}{\sum_{i=1}^{N} w_i}$$

4. Compute the learner’s weight:
$$\alpha_m = \frac{1}{2} \ln \left( \frac{1 - \varepsilon_m}{\varepsilon_m} \right)$$

5. Update sample weights:
$$w_i \leftarrow w_i \cdot e^{-\alpha_m y_i h_m(x_i)}$$

6. Normalize weights so that ∑wi=1.
7. FInal model : 
$$F(x) = \text{sign}\left( \sum_{m=1}^{M} \alpha_m h_m(x) \right)$$

2) XGBoost (Extreme Gradient Boosting)
XGBoost is an optimized implementation of Gradient Boosting that focuses on:

    Regularization to prevent overfitting
    Parallel computation
    Handling missing values
    Tree pruning and sparsity awareness

Objective function : 
$$\mathcal{L} = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{m=1}^{M} \Omega(f_m)$$

where 
Ω(fm) is the regularization term:
$$\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$

where,
    T: number of leaves
    wj: weight of leaf 
    γ: penalty for number of leaves
    λ: L2 regularization term

**Advantages**
✅ High predictive accuracy
✅ Works well with mixed-type data
✅ Handles complex non-linear relationships
✅ XGBoost offers regularization and scalability
✅ Performs feature selection implicitly

**Disadvantages**
❌ Computationally expensive
❌ Sensitive to noise and overfitting (if not regularized)
❌ Requires careful hyperparameter tuning
❌ Less interpretable compared to simple models

**Python Implementation (XGBoost)**
```python
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

![Gradient boosting sequential tree illustration](gb.png)

10. Evaluation Metrics
Evaluation metrics are quantitative measures used to assess the performance of a machine learning model — especially for classification tasks.
They help understand how well the model predicts true labels and handles errors.

These metrics are derived from a confusion matrix, which summarizes predictions vs. actual outcomes.

- Accuracy : Accuracy measures the overall correctness of the model — the proportion of total predictions that were correct.
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Interpretation:

    Indicates how often the model is correct.
    Works well when classes are balanced.
    Misleading for imbalanced datasets.

- Precision : Precision measures the accuracy of positive predictions — i.e., how many of the predicted positives are actually correct.
$$\text{Precision} = \frac{TP}{TP + FP}$$

Interpretation:
    High precision → model makes few false positive errors.
    Important when the cost of a false positive is high
(e.g., predicting someone has a disease when they don’t)

- Recall : Recall measures the ability of the model to detect all positive instances — i.e., how many actual positives are correctly predicted.
$$\text{Recall} = \frac{TP}{TP + FN}$$

Interpretation:

    High recall → model correctly identifies most positive cases.
    Important when the cost of a false negative is high
(e.g., missing a cancer diagnosis).

- F1-score : F1-score combines Precision and Recall into a single metric — balancing both.
$$F1 = 2 \times \frac{(\text{Precision} \times \text{Recall})}{(\text{Precision} + \text{Recall})}$$

Interpretation:

    High F1 → good balance between precision and recall.
    Preferred for imbalanced datasets where accuracy is unreliable.

- ROC : The ROC curve is a graphical plot showing the trade-off between:
    True Positive Rate (TPR) = Recall	​
    False Positive Rate (FPR)
Each point on the curve represents a model’s performance at a specific classification threshold.
	
-AUC: Area under ROC curve. AUC quantifies the overall ability of the model to discriminate between positive and negative classes.
$$\text{AUC} = \int_{0}^{1} TPR(FPR) \, d(FPR)$$

Interpretation:

    AUC = 1 → perfect classifier
    AUC = 0.5 → random guessing
    Higher AUC → better model discrimination capability

![ROC curve illustration](roc.png)

11. Regularization
------------------

Regularization is a technique to prevent overfitting by adding a penalty term to the loss function of a model. Overfitting occurs when a model learns the noise in the training data rather than the underlying pattern.

Mathematically, the regularized loss function is:

Loss_regularized = Loss_original + λ⋅Penalty

where:
λ = regularization parameter controlling the strength of the penalty

Penalty = depends on the type of regularization
- Ridge (L2), Lasso (L1), ElasticNet

- Ridge Regression (L2 Regularization)

   Adds a penalty proportional to the square of the coefficients.

   Loss function:

   $$\text{Loss}_{\text{Ridge}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$

   Effect:

   Shrinks coefficients towards zero, but rarely makes them exactly zero.

   Useful when all features are relevant but need smaller weights.

   Works well with multicollinearity (highly correlated features).

   Pros: Reduces variance, stable solution

   Cons: Does not perform feature selection

- Lasso Regression (L1 Regularization)

   Adds a penalty proportional to the absolute value of the coefficients.

   Loss function:

   $$\text{Loss}_{\text{Lasso}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$

   Effect:

   Can shrink some coefficients exactly to zero, performing feature selection.

   Useful when only a few features are important.

   Pros: Automatic feature selection

   Cons: Can struggle if features are highly correlated

- ElasticNet

   Combines Ridge (L2) and Lasso (L1) penalties.

   Loss function:

  $$\text{Loss}_{\text{ElasticNet}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{j=1}^{p} \beta_j^2$$

   Effect:

   Balances feature selection (from Lasso) and coefficient shrinkage (from Ridge).

   Particularly useful when there are many correlated features.

   Pros: Flexible, robust, works well in high-dimensional data

   Cons: Needs tuning of two hyperparameters (λ1 and λ2)

**Python Implementation**
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
```
