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

====================
UNSUPERVISED LEARNING
====================

Overview of Unsupervised Learning
------------------------------------
Unsupervised Learning deals with unlabeled data. The goal is to find hidden patterns, structures, or groupings in the data. Unlike supervised learning, there is no target variable.

Common tasks:
- Clustering: Grouping similar data points.
- Dimensionality Reduction: Reducing feature space while retaining important information.
- Anomaly Detection: Identifying unusual or rare events.

Workflow:
1. Data preprocessing
2. Model selection
3. Training on unlabeled data
4. Evaluating clusters or latent structure

![Graph: Unsupervised learning workflow](unsupervised_learning.webp)

Different tasks done in unsupervised learning:

1. Clustering
Clustering is a technique that groups similar data points together into clusters based on their characteristics, without using any labeled data. The objective is to ensure that data points within the same cluster are more similar to each other than to those in different clusters, enabling the discovery of natural groupings and hidden patterns in complex datasets.

### Different types of Clustering###
1. Types of Clustering

Let's see the types of clustering,

1. Hard Clustering: In hard clustering, each data point strictly belongs to exactly one cluster, no overlap is allowed. This approach assigns a clear membership, making it easier to interpret and use for definitive segmentation tasks.

    Example: If clustering customer data into 2 segments, each customer belongs fully to either Cluster 1 or Cluster 2 without partial memberships.
    Use cases: Market segmentation, customer grouping, document clustering.
    Limitations: Cannot represent ambiguity or overlap between groups; boundaries are crisp.

2. Soft Clustering: Soft clustering assigns each data point a probability or degree of membership to multiple clusters simultaneously, allowing data points to partially belong to several groups.

    Example: A data point may have a 70% membership in Cluster 1 and 30% in Cluster 2, reflecting uncertainty or overlap in group characteristics.
    Use cases: Situations with overlapping class boundaries, fuzzy categories like customer personas or medical diagnosis.
    Benefits: Captures ambiguity in data, models gradual transitions between clusters.

### Different types of Clustering Methods ###
1. Centroid-based Clustering (Partitioning Methods)
---------------------

**Introduction**
Centroid-based clustering organizes data points around central prototypes called centroids, where each cluster is represented by the mean (or medoid) of its members. The number of clusters is specified in advance and the algorithm allocates points to the nearest centroid, making this technique efficient for spherical and similarly sized clusters but sensitive to outliers and initialization.

**Algorithms**
There are two types of algorithms used for this kind of clustering method;
1. K-mean clustering : 
    K-Means is a centroid-based clustering algorithm that partitions n observations into k clusters in which each observation belongs to the cluster with the nearest mean.

    **Algorithm**
    1. Choose number of clusters k.
    2. Initialize k centroids randomly.
    3. Assign each point to nearest centroid.
    4. Update centroids as mean of assigned points.
    5. Repeat steps 3-4 until convergence.

    **Mathematical Formulation**
    Minimize within-cluster sum of squares (WCSS):
        J = Σ Σ ||x_i - μ_j||^2
    where μ_j is the centroid of cluster j.

    **Python Implementation**
    ```python
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X)
    y_pred = model.labels_
    ```
    **Advantages**
        - Easy to understand and implement: K-means is a straightforward algorithm that is simple to grasp and code, especially with pre-built libraries.
        - Interpretability: It is easy to interpret the results of k-means clustering

    **Disadvantages**
        - Choosing the Right Number of Clusters (kk): One of the biggest challenges is deciding how many clusters to use.
        - Sensitive to Initial Centroids: The final clusters can vary depending on the initial random placement of centroids.
        - Non-Spherical Clusters: K-Means assumes that the clusters are spherical and equally sized. This can be a problem when the actual clusters in the data are of different shapes or densities.
        - Outliers: K-Means is sensitive to outliers, which can distort the centroid and, ultimately, the clusters.

2. K-Medoids clustering : 
    K-Medoids, also known as Partitioning Around Medoids (PAM), is a clustering algorithm introduced by Kaufman and Rousseeuw. It is similar to K-Means, but instead of using the mean of points as a cluster center, it uses an actual data point called a medoid.

    Medoids - A medoid is the most centrally located data point within a cluster. It minimizes the total dissimilarity with all other points in that cluster. The dissimilarity between a medoid Ci and an object Pi​ is given by: E=∣Pi−Ci∣

    The total cost (or objective function) of K-Medoids is defined as:
    $$c = \frac{\sum C_i}{\sum_{P_i \in C_i} |P_i - C_i|}$$

    **Algorithm**
    1. Randomly select k data points from the dataset as initial medoids.
    2. Assign each data point to the nearest medoid using a distance metric (e.g., Manhattan or Euclidean).
    3. For each medoid m, try swapping it with a non-medoid point ooo.
        - Recalculate the cost for this new configuration.
        - If the total cost decreases, accept the swap; otherwise, revert.
    4. Continue until no further cost reduction is possible.

    **Python Implementation**
    ```python
    from sklearn_extra.cluster import KMedoids
    from sklearn.datasets import make_blobs
    model = KMedoids(n_clusters=3, random_state=42)
    model.fit(X)
    y_pred = model.labels_
    ```
    **Advantages**
        - It is simple to understand and easy to implement.
        - K-Medoids converges in a fixed number of steps.
        - It is less sensitive to outliers compared to other partitioning algorithms.

    **Disadvantages**
        - Not suitable for non-spherical or arbitrarily shaped clusters.
        - Results may differ across runs due to random initialization of medoids.

2. Hierarchical Clustering
---------------------------
**Introduction**
Hierarchical Clustering is an unsupervised learning method used to group similar data points into clusters based on their distance or similarity. Instead of choosing the number of clusters in advance, it builds a tree-like structure called a dendrogram that shows how clusters merge or split at different levels. It helps identify natural groupings in data and is commonly used in pattern recognition, customer segmentation, gene analysis and image grouping.

**Types of Hierarchical Clustering**
There are two main types of hierarchical clustering.

1. Agglomerative (Bottom-top) Approach : 
    It is also known as the bottom-up approach or hierarchical agglomerative clustering (HAC). Bottom-up algorithms treat each data as a singleton cluster at the outset and then successively agglomerate pairs of clusters until all clusters have been merged into a single cluster that contains all data.

**Algorithm (Agglomerative)**
1. Start with individual points: Each data point is its own cluster. For example if we have 5 data points we start with 5 clusters each containing just one data point.
2. Calculate distances between clusters: Calculate the distance between every pair of clusters. Initially since each cluster has one point this is the distance between the two data points.
3. Merge the closest clusters: Identify the two clusters with the smallest distance and merge them into a single cluster.
4. Update distance matrix: After merging we now have one less cluster. Recalculate the distances between the new cluster and the remaining clusters.
5. Repeat steps 3 and 4: Keep merging the closest clusters and updating the distance matrix until we have only one cluster left.
6. Create a dendrogram: As the process continues we can visualize the merging of clusters using a tree-like diagram called a dendrogram. It shows the hierarchy of how clusters are merged.

![ Agglomerative Approach](agglomerative.png)

**Linkage Criteria**
Key part of this process is linkage which calculates the distance between clusters before they are merged or divided. Different types of linkage is used measure this distance differently.

1. Single Linkage

For two clusters R and S the single linkage returns the minimum distance between two points. This method creates long, chain-like clusters because it is sensitive to outliers and can connect clusters based on a very small number of close points.

$$L(R, S) = \min\bigl(D(i, j)\bigr),\; i \in R,\; j \in S$$
where
D(i, j): Distance function between points i and j.

![Single Linkage](Single-Linkage.jpg)

2. Complete Linkage

For two clusters R and S the complete linkage returns the maximum distance between two points. It tends to create compact and spherical clusters because it is more sensitive to outliers and tries to make sure that the clusters are not too far.

$$L(R, S) = \max\bigl(D(i, j)\bigr),\; i \in R,\; j \in S$$

![Complete Linkage](complete-Linkage.jpg)

3. Average Linkage

It returns the average distance between all pairs of points from two clusters. This method maintain a balance between single and complete linkage by considering all pairs of points not just the closest or farthest point. It usually results in clusters that are moderately compact.

$$L(R, S) = \frac{1}{n_R \times n_S} \sum_{i=1}^{n_R} \sum_{j=1}^{n_S} D(i, j),\; i \in R,\; j \in S$$
where
n_R​ : Number of data-points in R
n_S​ : Number of data-points in S

![Average Linkage](average.png)

4. Ward's Linkage

It calculates the distance between two clusters by looking at total spread or variance increase when the clusters are combined. This method creates compact, well-separated clusters by making sure that data within each cluster is as similar as possible.

$$L(R, S) = {n_R + n_S}\frac{n_R \times n_S}
\sum_{i=1}^{n_R} \sum_{j=1}^{n_S} D(i, j),\; i \in R,\; j \in S
$$
where
nR​​ and nS​ are the sizes of clusters R and S
D(i, j) is the distance between points i∈R and j∈S.

![Ward's Linkage](ward-Linkage.jpg)

5. Centroid Linkage

It calculates the distance between two clusters based on the distance between their central points i.e the average of all points in the cluster. This method works well when clusters are round or evenly shaped but it may not be the best for irregularly shaped clusters.

$$L(R, S) = D(\overline{R}, \overline{S})$$

where
$$\overline{R}$$ and $$\overline{S}$$ are the centroids (mean points) of clusters R and S
$$D(\overline{R},\overline{S})$$ is the distance between the centroids of clusters R and S.

![Centroid Linkage](Centroid-Linkage.jpg)

**Python Implementation**
```python
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_pred = model.fit_predict(X)
```



3. DBSCAN (Density-Based Spatial Clustering)
-------------------------------------------
**Introduction**
DBSCAN is a density-based clustering algorithm that groups data points that are closely packed together and marks outliers as noise based on their density in the feature space. It identifies clusters as dense regions in the data space separated by areas of lower density. Unlike K-Means or hierarchical clustering which assumes clusters are compact and spherical, DBSCAN perform well in handling real-world data irregularities such as:

    - Arbitrary-Shaped Clusters: Clusters can take any shape not just circular or convex.
    - Noise and Outliers: It effectively identifies and handles noise points without assigning them to any cluster.

**Parameters**
- eps: This defines the radius of the neighborhood around a data point. If the distance between two points is less than or equal to eps they are considered neighbors. A common method to determine eps is by analyzing the k-distance graph. Choosing the right eps is important:

    - If eps is too small most points will be classified as noise.
    - If eps is too large clusters may merge and the algorithm may fail to distinguish between them.

- MinPts: This is the minimum number of points required within the eps radius to form a dense region. A general rule of thumb is to set MinPts >= D+1 where D is the number of dimensions in the dataset.

    - For most cases a minimum value of MinPts = 3 is recommended.

**Algorithm**
1. Identify Core Points: For each point in the dataset count the number of points within its eps neighborhood. If the count meets or exceeds MinPts mark the point as a core point.
2. Form Clusters: For each core point that is not already assigned to a cluster create a new cluster. Recursively find all density-connected points i.e points within the eps radius of the core point and add them to the cluster.
3. Density Connectivity: Two points a and b are density-connected if there exists a chain of points where each point is within the eps radius of the next and at least one point in the chain is a core point. This chaining process ensures that all points in a cluster are connected through a series of dense regions.
4. Label Noise Points: After processing all points any point that does not belong to a cluster is labeled as noise.

**Python Implementation**
```python
from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.5, min_samples=5) #min_samples = MinPts here
y_pred = model.fit_predict(X)
```

2. Dimensionality Reduction :
When working with machine learning models, datasets with too many features can cause issues like slow computation and overfitting. Dimensionality reduction helps to reduce the number of features while retaining key information. It converts high-dimensional data into a lower-dimensional space while preserving important details.
For example, when you are building a model to predict house prices with features like bedrooms, square footage and location. If you add too many features such as room condition or flooring type, the dataset becomes large and complex.

How Dimensionality Reduction Works?

Lets understand how dimensionality Reduction is used with the help of example. Imagine a dataset where each data point exists in a 3D space defined by axes X, Y and Z. If most of the data variance occurs along X and Y then the Z-dimension may contribute very little to understanding the structure of the data.

![Dimensions](dimension.webp)

Before Reduction we can see that data exist in 3D (X,Y,Z). It has high redundancy and Z contributes little meaningful information
On the right after reducing the dimensionality the data is represented in lower-dimensional spaces. The top plot (X-Y) maintains the meaningful structure while the bottom plot (Z-Y) shows that the Z-dimension contributed little useful information. 

This process makes data analysis more efficient hence improving computation speed and visualization while minimizing redundancy

**Dimensionality Reduction techniqies can be broadly devided into two main categories:

1. Feature Selection

Feature selection chooses the most relevant features from the dataset without altering them. It helps remove redundant or irrelevant features, improving model efficiency. Some common methods are:

    - Filter methods rank the features based on their relevance to the target variable.
    - Wrapper methods use the model performance as the criteria for selecting features.
    - Embedded methods combine feature selection with the model training process.

2. Feature Extraction

Feature extraction involves creating new features by combining or transforming the original features. These new features retain most of the dataset’s important information in fewer dimensions. Common feature extraction methods are:

1. Principal Component Analysis (PCA)
-------------------------------------
**Introduction**
PCA (Principal Component Analysis) is a dimensionality reduction technique and helps us to reduce the number of features in a dataset while keeping the most important information. It changes complex datasets by transforming correlated features into a smaller set of uncorrelated components.

**Working**
PCA uses linear algebra to transform data into new features called principal components. It finds these by calculating eigenvectors (directions) and eigenvalues (importance) from the covariance matrix. PCA selects the top components with the highest eigenvalues and projects the data onto them simplify the dataset.

Step 1: Standardize the Data
Different features may have different units and scales like salary vs. age. To compare them fairly PCA first standardizes the data by making each feature have:

A mean of 0
A standard deviation of 1

$$Z = \frac{X - \mu}{\sigma}$$
where:
μ is the mean of independent features  μ={μ1,μ2,⋯ ,μm}.
σ is the standard deviation of independent features  σ={σ1,σ2,⋯ ,σm}.

Step 2: Calculate Covariance Matrix

Next PCA calculates the covariance matrix to see how features relate to each other whether they increase or decrease together. The covariance between two features x1x1​ and x2x2​ is:

$$\operatorname{cov}(x_1, x_2)
= \frac{1}{n - 1} \sum_{i=1}^{n} (x_{1i} - \bar{x}_1)(x_{2i} - \bar{x}_2)
$$

Where:
\bar{x}_1 and \bar{x}_2​ are the mean values of features x1 and x2.
n is the number of data points.

The value of covariance can be positive, negative or zeros.

Step 3: Find the Principal Components

PCA identifies new axes where the data spreads out the most:

    - 1st Principal Component (PC1): The direction of maximum variance (most spread).
    - 2nd Principal Component (PC2): The next best direction, perpendicular to PC1 and so on.

These directions come from the eigenvectors of the covariance matrix and their importance is measured by eigenvalues. For a square matrix A an eigenvector X (a non-zero vector) and its corresponding eigenvalue λ satisfy:

    AX=λX

This means:
When A acts on X it only stretches or shrinks X by the scalar λ.
The direction of X remains unchanged hence eigenvectors define "stable directions" of A.

Eigenvalues help rank these directions by importance.

Step 4: Pick the Top Directions & Transform Data

After calculating the eigenvalues and eigenvectors PCA ranks them by the amount of information they capture. We then:

    - Select the top k components that capture most of the variance like 95%.
    - Transform the original dataset by projecting it onto these top components.

This means we reduce the number of features (dimensions) while keeping the important patterns in the data.

**Python Implementation**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```
**Advantages**

    - Multicollinearity Handling: Creates new, uncorrelated variables to address issues when original features are highly correlated.
    - Noise Reduction: Eliminates components with low variance enhance data clarity.
    - Data Compression: Represents data with fewer components reduce storage needs and speeding up processing.
    - Outlier Detection: Identifies unusual data points by showing which ones deviate significantly in the reduced space.

**Disadvantages**

    - Interpretation Challenges: The new components are combinations of original variables which can be hard to explain.
    - Data Scaling Sensitivity: Requires proper scaling of data before application or results may be misleading.
    - Information Loss: Reducing dimensions may lose some important information if too few components are kept.
    - Assumption of Linearity: Works best when relationships between variables are linear and may struggle with non-linear data.
    - Computational Complexity: Can be slow and resource-intensive on very large datasets.
    - Risk of Overfitting: Using too many components or working with a small dataset might lead to models that don't generalize well.

2. Independent Component Analysis (ICA)
----------------------------------------
**Introduction**
Independent Component Analysis (ICA) is a technique used to separate mixed signals into their independent, non-Gaussian components. Its aim to find a linear transformation of data that maximizes statistical independence among the components. ICA is widely used in fields like audio, image processing and biomedical signal analysis to isolate distinct sources from mixed signals.

Assumptions in ICA

ICA operates under two key assumptions:

    - The source signals are statistically independent of each other.
    - The source signals have non-Gaussian distributions.

These assumptions allow ICA to effectively separate mixed signals into independent components, a task that traditional methods like PCA cannot achieve

Cocktail Party Problem in ICA

To better understand how Independent Component Analysis (ICA) works let’s look at a classic example known as the Cocktail Party Problem
ICA_Problem
Cocktail Party Problem in ICA

Here there is a party going into a room full of people.

There is 'n' number of speakers in that room and they are speaking simultaneously at the party. In the same room, there are also 'n' microphones placed at different distances from the speakers which are recording 'n' speakers' voice signals. 

![Cocktail Part example for ICA](ica.webp)

Hence the number of speakers is equal to the number of microphones in the room. Now using these microphones' recordings, we want to separate all the 'n' speakers voice signals in the room given that each microphone recorded the voice signals coming from each speaker of different intensity due to the difference in distances between them.

Decomposing the mixed signal of each microphone's recording into an independent source's speech signal can be done by using the machine learning technique independent component analysis. 

     [X1,X2,…,Xn]=>[Y1,Y2,…,Yn]

where X1,X2,…,Xn​ are the original signals present in the mixed signal and Y1,Y2,…,Yn​ are the new features and are independent components that are independent of each other.

**Applications**
- Blind source separation (e.g., separating audio sources)
- Feature extraction


**Python Implementation**
```python
from sklearn.decomposition import FastICA
ica = FastICA(n_components=2)
X_independent = ica.fit_transform(X)
```
**Advantages**

    - Separation of Mixed Signals: ICA is a go-to tool for separating mixed signals into their independent components. This is useful in a variety of applications such as signal processing, image analysis and data compression.
    - Non-Parametric Approach: It is a non-parametric approach which means that it does not require assumptions about the underlying probability distribution of the data.
    - Unsupervised Learning Technique: It is an unsupervised learning technique which means that it can be applied to data without the need for labeled examples. This makes it useful in situations where labeled data is not available.
    - Useful for Feature Extraction: This can be used for feature extraction which means that it can identify important features in the data that can be used for other tasks, such as classification.

**Disadvantages**

    - Assumes Non-Gaussian Sources: It assumes that the underlying sources are non-Gaussian which may not always be true. If the underlying sources are Gaussian ICA may not be effective.
    - Assumes Linear Mixing: ICA assumes that the sources are mixed linearly which may not always be the case. If the sources are mixed nonlinearly ICA may not be effective.
    - Computationally Expensive: This can be computationally expensive especially for large datasets which make it difficult to apply ICA to real-world problems.

3. t-Distributed Stochastic Neighbor Embedding (t-SNE)
------------------------------------------------------
**Introduction**
T-distributed Stochastic Neighbor Embedding (t-SNE) is a non linear dimensionality reduction technique used for visualizing high-dimensional data in a lower-dimensional space mainly in 2D or 3D. Unlike linear methods such as Principal Component Analysis (PCA), t-SNE focus on preserving the local structure and pattern of the data. 
t-SNE works by looking at the similarity between data points in the high-dimensional space. The similarity is computed as a conditional probability. It calculates how likely it is that one data point would be near another.

**Algorithm**
- Compute pairwise similarities in high-dimensional space.
- Map points to lower-dimensional space preserving similarities.
- Minimize Kullback-Leibler divergence.

**Python Implementation**
```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X)
```

**Advantages**

    - Great for Visualization: t-SNE is particularly used to convert complex high-dimensional data into 2D or 3D for visualization making patterns and clusters easy to observe.
    - Preserve Local Structure: Unlike linear techniques like PCA t-SNE focus on maintaining the local relationships between data points meaning similar data points remain close in the lower-dimensional space.
    - Non-Linear Capability: It captures non-linear dependencies in the data which makes it suitable for complex datasets where linear methods fail.
    - Cluster Separation: Helps in clearly visualizing clusters and class separability in datasets like MNIST making it easier for interpretation and exploration.

**Disadvantages**

    - Computationally Intensive: t-SNE is slower and more computationally expensive compared to linear methods especially on large datasets.
    - Non-deterministic Output: The output can vary with each run due to its randomness unless a fixed random_state is used.
    - Not Scalable for Large Datasets: It struggles with very large datasets (e.g., millions of points) unless optimized or approximated versions are used.
    - Not Good for Downstream Tasks: t-SNE is mainly for visualization and is not suitable for dimensionality reduction when feeding data into other ML algorithms.
    - No Global Structure Preservation: It may distort global distances and structures in the data focusing more on preserving local neighborhoods.

**Evaluation Metrics for Clustering**
------------------------------------
- Silhouette Score: 
    The Silhouette Score is a way to measure how good the clusters are in a dataset. It helps us understand how well the data points have been grouped. The score ranges from -1 to 1.

    - A score close to 1 means a point fits really well in its group (cluster) and is far from other groups.
    - A score close to 0 means the point is on the border between two clusters.
    - A score close to -1 means the point might be in the wrong cluster.

    Silhouette Score (S) for a data point i is calculated as:
    $$S(i) = \frac{b(i) - a(i)}{\max(a(i),\, b(i))}$$

    where,
    a(i) is the average distance from i to other data points in the same cluster.
    b(i) is the smallest average distance from i to data points in a different cluster.

- Davies-Bouldin Index

    The Davies-Bouldin Index (DBI) helps us measure how good the clustering is in a dataset. It looks at how tight each cluster is (compactness), and how far apart the clusters are (separation).

        - Lower DBI = better, clearer clusters
        - Higher DBI = messy, overlapping clusters

    A lower score is better, because it means:

        - Points in the same cluster are close to each other.
        - Different clusters are far apart from one another.

    Davies-Bouldin Index (DB) is calculated as:
$$DB = \frac{1}{k} \sum_{i=1}^{k} 
\max_{j \ne i} \left( \frac{R_{ij}}{R_{ii} + R_{jj}} \right)$$

where,

    k is the total number of clusters.
    $$R_{ii}$$ is the compactness of cluster i.
    $$R_{ii}$$ is the compactness of cluster j.
    $$R_{ii}R_{jj}$$​ is the dissimilarity (distance) between cluster i and cluster j.

3. Calinski-Harabasz Index (Variance Ratio Criterion)

    The Calinski-Harabasz Index measures how good the clusters are in a dataset.

    It looks at:

        How close the points are inside each cluster?
        How far apart the clusters are?

    A higher score is better, as it means the clusters are tight and well-separated. It helps determine the ideal number of clusters.

    Calinski-Harabasz Index (CH) is calculated as:
    $$CH = \frac{W_B}{N - K} \times (K - 1)$$

where,

    B is the sum of squares between clusters.
    W is the sum of squares within clusters.
    N is the total number of data points.
    K is the number of clusters.

    Calculating between group sum of squares (B):
$$W = \sum_{k=1}^{n_k} \lVert X_{ik} - C_k \rVert^2$$
where,

    nk​ is the number of observation in cluster 'k'
    Xik is the i-th observation of cluster 'k'
    Ck​ is the centroid of cluster 'k'

    Calculating within the group sum of squares (W)

$$W = \sum_{k=1}^{K} \sum_{i=1}^{n_k} \lVert X_{ik} - C_k \rVert^2$$

where,

    nk​ is the number of observation in cluster 'k'
    Xik is the i-th observation of cluster 'k'
    Ck is the centroid of cluster 'k'

4. Adjusted Rand Index (ARI)

    The Adjusted Rand Index (ARI) helps us measure how accurate a clustering result is by comparing it to the true labels (ground truth).

    It checks how well the pairs of points are grouped:

        - Are the same pairs together in both the real and predicted clusters?
        Are different pairs also kept apart correctly?

    The score ranges from -1 to 1:

        - 1 means perfect match - the clustering is exactly right.
        - 0 means random guess - no better than chance.
        - Below 0 means worse than random - very poor clustering.

    Adjusted Rand Index (ARI) is calculated as:
    $$ARI = \frac{RI - \text{ExpectedRI}}{\max(RI) - \text{ExpectedRI}}$$

where,

    RI is the Rand Index.
    ExpectedRIExpectedRI​ is the expected value of the Rand Index.

5. Mutual Information (MI)

Mutual Information measures how much two variables are related or connected. In clustering, it compares how much the true cluster labels match with the predicted labels. It shows how much knowing about one variable helps us predict the other. The more agreement there is, the higher the score.

    - Higher values mean better agreement between the clusters.
    - Zero means no agreement at all.

MI between true labels Y and predicted labels Z is calculated as:
$$MI(y, z) = \sum_i \sum_j p(y_i, z_j)\,\log\!\left( \frac{p(y_i)\,p(z_j)}{p(y_i, z_j)} \right)
$$

where,

    yi​ is a true label.
    zi​ is a predicted label.
    p(yi,zi) is the joint probability of yi​ and zj​.
    p(yi) and p(zi) are the marginal probabilities.

These clustering metrics help in evaluating the quality and performance of clustering algorithms, allowing for informed decisions when selecting the most suitable clustering solution for a given dataset.

**Python Implementation**
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score,  mutual_info_score, adjusted_rand_score

model = KMeans(n_clusters=3)
model.fit(X)

silhouette = silhouette_score(X, model.labels_)
db_index = davies_bouldin_score(X, model.labels_)
ch_index = calinski_harabasz_score(X, model.labels_)
ari = adjusted_rand_score(iris.target, model.labels_)
mi = mutual_info_score(iris.target, model.labels_)
```

====================
REINFORCEMENT LEARNING
====================

Overview of Reinforcement Learning (RL)
-----------------------------------------
Reinforcement Learning is a learning paradigm where an agent learns to make sequential decisions by interacting with an environment to maximize long-term cumulative reward.

The agent does not receive labeled data.
Instead, it receives rewards, which guide it toward good behavior.

Key components:
- Agent: Learner or decision maker
- Environment: The world the agent interacts with
- State (S): Current situation of the agent
- Action (A): Choices available to the agent
- Reward (R): Feedback signal
- Policy (π): Strategy mapping states to actions
- Value Function (V): Expected cumulative reward
- Q-Function (Q): Expected reward for state-action pair

![Reinforcement Learning](reinforcement_learning.jpg)

1. RL as a Markov Decision Process (MDP)

RL problems are modeled as a Markov Decision Process (MDP) defined by:
- Tuple (S, A, P, R, γ)
  - S: States
  - A: Actions
  - P: Transition probabilities P(s'|s,a)
  - R: Reward function
  - γ: Discount factor (0 <= γ <= 1)

2. Return (Cumulative Discounted Reward)

The return from time step t is:
    $G_t = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1}$

This represents the total future reward, discounted by γ.

3. Policy

A policy defines the agent’s behavior:$\pi(a|s) = \Pr(A_t = a \mid S_t = s)$

4. Value Functions (Core of RL)
Focus on estimating “how good” a state or action is.

    4.1 State-Value Function
$V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right]$

    4.2 Action-Value Function (Q-function)
    - Model-free algorithm to learn Q-function
    - Update rule:

$Q^\pi(s,a) = \mathbb{E}_\pi \left[ G_t \mid S_t = s, A_t = a \right]$

5. Bellman Equations
    5.1 Bellman Expectation Equation for $V^\pi$
    $V^\pi(s) = \sum_{a}\pi(a|s)\sum_{s'}P(s'|s,a)\left[R(s,a) + \gamma V^\pi(s')\right]$

    5.2 Bellman Expectation Equation for $Q^\pi$
    $Q^\pi(s,a) = \sum_{s'}P(s'|s,a)\left[R(s,a) + \gamma \sum_{a'}\pi(a'|s')Q^\pi(s',a')\right]$

6. Optimal Value Functions
$$V^*(s) = \max_\pi V^\pi(s)
Q^*(s,a) = \max_\pi Q^\pi(s,a)
$$

7. Bellman Optimality Equations
7.1 For $V^*(s)$
$V^*(s) = \max_{a}\sum_{s'}P(s'|s,a)\left[R(s,a) + \gamma V^*(s')\right]$

7.2 For $Q^*(s,a)$
$Q^*(s,a) = \sum_{s'}P(s'|s,a)\left[R(s,a) + \gamma \max_{a'}Q^*(s',a')\right]$

8. Major RL Algorithms
    8.1 Dynamic Programming (DP)

    Requires full knowledge of transition probabilities.

    Value Iteration
    $V_{k+1}(s) = \max_a \sum_{s'}P(s'|s,a)\left[R(s,a)+\gamma V_k(s')\right]$

    Policy Iteration
        1.Policy evaluation
        2.Policy improvement

    8.2 Monte Carlo (MC)

    Learns value functions from sampled episodes.

    Update rule: $V(s) \leftarrow V(s) + \alpha \left[G_t - V(s)\right]$

    8.3 Temporal Difference (TD) Learning

    Combines DP + MC.

    TD(0) Update Rule : $V(s) \leftarrow V(s) + \alpha \left[R + \gamma V(s') - V(s)\right]$

    8.4 Q-Learning (Off-Policy):
        A value-based, model-free, off-policy method.
        Learns which action is best in each state.
        Agent gradually improves its estimate of the best behavior.

    One of the most important RL algorithms.
    $Q(s,a) \leftarrow Q(s,a) + \alpha\left[R + \gamma \max_{a'}Q(s',a') - Q(s,a)\right]$

    8.5 SARSA (On-Policy):
        Similar to Q-learning but on-policy.
        Learns values based on the actions it actually takes.

    $Q(s,a) \leftarrow Q(s,a) + \alpha\left[R + \gamma Q(s',a') - Q(s,a)\right]$

9. Exploration vs Exploitation

Exploration : Trying different actions to discover new possibilities.

Exploitation : Choosing the best-known action to get maximum reward.

RL requires a balance of both:

    Too much exploration → slow progress
    Too much exploitation → agent may miss better strategies

    - ε-greedy strategy: choose random action with probability ε, otherwise best action
    - Softmax or Boltzmann exploration

ϵ-Greedy Policy : 
$\pi(a|s) =
\begin{cases}
1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|}, & \text{if } a = \arg\max_{a'}Q(s,a') \\
\frac{\epsilon}{|\mathcal{A}|}, & \text{otherwise}
\end{cases}
$

10. Policy Gradient Methods

Instead of learning value functions and deriving a policy, policy gradient methods directly optimize the policy.

Objective: $J(\theta) = \mathbb{E}_\pi\left[ G_t\right]$

Policy Gradient Theorem : 
$\nabla_\theta J(\theta)
= \mathbb{E}_\pi \left[ G_t \nabla_\theta \log \pi_\theta(a|s) \right]$

This leads to REINFORCE algorithm.

11. Actor–Critic Methods
    Combines value-based and policy-based approaches

Combine:
    Actor → updates policy
    Critic → evaluates action via value function

Update: $\theta \leftarrow \theta + \alpha\, \delta \nabla_\theta \log \pi_\theta(a|s$
Where TD error: $\delta = R + \gamma V(s') - V(s)$

12. Model-Free vs Model-Based RL
    Model-Free RL
        Learns only from experience.
        Does not try to understand how the environment works.

    Examples:
        Q-learning
        DQN
        PPO
        A3C

    Model-Based RL
        Learns a model of the environment and uses planning.

    Examples:
        AlphaZero-style methods

**Python Implementation (simplified)**
```python
import numpy as np
Q = np.zeros((num_states, num_actions))
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = choose_action(Q, state)
        next_state, reward, done = env.step(action)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
```

**Applications**
----------------
- Robotics: Path planning, manipulation
- Game AI: Chess, Go, video games
- Recommendation systems
- Autonomous driving

*Advantages of RL*
    Learns sequential decision-making
    Works with delayed rewards
    Can outperform humans on difficult tasks
    Handles continuous control
    Works without labeled datasets

10. Limitations of RL
    Requires large training time
    Highly sensitive to hyperparameters
    Exploration can be difficult
    Not guaranteed to converge
    Environments must be well simulated

====================
END OF REINFORCEMENT LEARNING SECTION
====================

