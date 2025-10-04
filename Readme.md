                                                     **TASK-1**

                                   **Term Deposit Subscription Prediction (Bank Marketing)**

**1.Install Required Packages**

Installs essential libraries: pandas, numpy, scikit-learn, matplotlib, shap, joblib.



**2.Import Libraries**

Loads Python packages for data handling, visualization, and machine learning.



**3.Load Dataset**

Reads the dataset (bank-additional-full.csv) into a DataFrame.

Creates a new binary target column subscribed from the original y column.



**4.Data Preprocessing**

Drops the duration column (to simulate model usability before call duration is known).

Encodes categorical variables using dummy variables (one-hot encoding).

Splits the dataset into features (X) and target (y).



**5.Train/Test Split**

Divides the dataset into training and testing sets (75/25 split, stratified).



**6.Model Training**

Trains two classifiers:

-Logistic Regression

-Random Forest Classifier



**7.Model Evaluation**

Evaluates models using:

-Confusion Matrix

-F1 Score

-ROC AUC Score

-ROC Curve Visualization

-Classification Report



**8.Feature Importance**

Uses Permutation Importance to identify key features.

Explains model behavior with SHAP (SHapley Additive exPlanations).

Explain Predictions with SHAP

Generates explanations for 5 random test samples to visualize how features impact predictions.



**9.Save Models**

Saves trained models (RandomForest and LogisticRegression) with joblib for reuse.



**10.Utilities \& Inspection**

Allows retrieval of original test sample rows from the dataset for better interpretability.





&nbsp;                                                 **TASK-2**

                                Customer Segmentation Using Unsupervised Learning



**1.Key Tasks Performed**

Library Imports: Imported tools for clustering, scaling, and visualization, including KMeans, StandardScaler, PCA, and TSNE.

Data Loading: Loaded the customer dataset (implied from the output of columns like CustomerID, Annual Income (k$), and Spending Score (1-100)).



**2.Preprocessing:**

The features used for clustering were likely scaled using StandardScaler to normalize their range.



**3.Clustering:**

The K-Means clustering algorithm was used to group customers into distinct segments.

The optimal number of clusters (K) was determined (though the elbow method or silhouette score may have been used, the final use of KMeans is confirmed).

A new column, Cluster, was added to the dataset to label each customer's segment.



**4.Dimensionality Reduction and Visualization:**

Principal Component Analysis (PCA) was performed to reduce the data to two dimensions for simple 2D visualization of the clusters.

t-distributed Stochastic Neighbor Embedding (t-SNE) was also set up (or executed) as an alternative, non-linear method to visualize better separation between the customer segments.



**6.Strategy Proposal:** 

The notebook's main goal is to use the resulting clusters to propose marketing strategies tailored to each segment (this is the final, non-coding task derived from the clustering results).





                                                         **TASK-4**

                                       **Loan Default Risk with Business Cost Optimization**

**1.Data Loading and Initial EDA:**

Loaded the large credit application dataset (shape: 307,511 rows, 122 columns).

Identified the target variable, TARGET (1 = default, 0 = non-default).

Performed initial Exploratory Data Analysis (EDA), noting the severe class imbalance (approximately 8% default rate).

Analyzed missing values and prepared to filter out columns with high missingness (e.g., columns with up to ∼70% missing data).



**2.Feature Preprocessing:**

Feature Selection: Filtered features based on missingness (numeric features with <60% missing) and low cardinality (categorical features with ≤50 unique values).



3.Data Transformation: Constructed a preprocessing pipeline using ColumnTransformer:

Numeric Features: Imputed missing values with the median and scaled features using StandardScaler.

Categorical Features: Imputed missing values with a constant ("MISSING") and applied One-Hot Encoding.



**4.Model Training:**

The processed data was split into training and testing sets (80% train, 20% test) using stratified sampling to preserve the target distribution.

Two models were trained and evaluated:

-Logistic Regression

-A Tree Model (either CatBoost or RandomForestClassifier as a fallback).

Initial performance was measured using Area Under the ROC Curve (AUC), yielding scores around 0.74.



**5.Cost-Benefit Threshold Optimization:**

Defined the core business costs: False Positive (FP) Cost (lost profit from rejecting a good customer, e.g., $1,000) and False Negative (FN) Cost (credit loss from accepting a defaulting customer, e.g., $20,000).

Implemented a function to calculate the total business cost Cost 

Total

=(FP×Cost 

FP

&nbsp;)+(FN×Cost 

FN

&nbsp;) for every possible classification threshold.

Optimized the threshold for both models to find the point that minimizes the calculated total business cost.



**6.Model Evaluation at Best Threshold:**

Visualized the relationship between the decision threshold and the total business cost.

Reported the final performance of the chosen model (Random Forest/Tree Model) at its optimal cost-based threshold (e.g., 0.066 for the example costs), including the Classification Report and Confusion Matrix.



**7.Feature Importance:** 

Calculated and displayed the Feature Importances for the tree model (Random Forest) to highlight the most predictive variables for default risk, such as EXT\_SOURCE\_2, EXT\_SOURCE\_3, and DAYS\_BIRTH.



**8.Sensitivity Analysis:** 

Performed a sensitivity analysis by testing a grid of nine different combinations of COST\_FP and COST\_FN values to understand how the optimal best threshold shifts under various business conditions. 

