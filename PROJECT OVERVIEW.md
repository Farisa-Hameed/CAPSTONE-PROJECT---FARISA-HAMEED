# HEALTHCARE DATASET - EVALUATION OF TEST RESULTS 

In this project, we aim to leverage machine learning techniques to predict outcomes based on features derived from a dataset, specifically in a classification context. The core objective is to build a robust, accurate, and scalable model that can predict a target variable (in this case, Test Results) based on a set of input features. Here, the Test Results represents classification scenario by  diagnosing diseases whether it is normal, abnormal, or inconclusive.
By building a model that can accurately predict a target variable based on multiple input features, we are taking a significant step toward making real-time decisions based on historical data. This capability can be a game-changer in business strategies, health diagnostics, and various other domains.

Various process Done includes: 

Data Loading and Initial Exploration:
Loading the Dataset: The dataset was loaded into a pandas DataFrame using pd.read_csv().
Initial Exploration: The basic structure of the data was examined using data.head() to display the first few rows of the dataset, and data.info() to gather information about the columns and their data types.

Data Preprocessing:
Handling Missing Values: There were no missing data. But found duplicates and removed them.
Checking Skewness: We checked the skewness of numerical features using data.skew(). No skewness found.
(Skewness values above 0.5 or below -0.5 typically indicate that the feature is highly skewed and may need transformation.)
Identifying Outliers:
Outliers in the data were checked using visualization techniques like boxplots and IQR. No Outliers found.

Exploratory Data Analysis (EDA):
Histograms of Numerical Columns: Histograms were plotted for the numerical columns to understand the distribution of data. This visualization helps identify features that are normally distributed, skewed, or have outliers.

Correlation Heatmap: A correlation matrix heatmap was plotted to visualize the relationships between numerical features. This helps identify strong correlations, which can aid in understanding how different features interact with each other, or in feature engineering.

Bar Plot for Categorical Column (Medical Condition): A bar plot was created to visualize the distribution of different medical conditions within the dataset.

Line Plot for Time-Series Data (e.g., 'Days Hospitalized' vs. 'Billing Amount'): A line plot was drawn to observe the trend over time between Days Hospitalized and Billing Amount. This kind of plot helps in visualizing any temporal relationships or trends between variables.

Pie Chart for Distribution of Medical Condition: A pie chart was used to show the proportion of different medical conditions in the dataset. This visualization helps in quickly understanding the class distribution of a categorical variable.

Kernel Density Estimation (KDE) Plot for Age: The KDE plot was created to understand the distribution of Age in the dataset. 

Feature Engineering:
We defined the features (X) and the target variable (y) for training and testing.
- X (features): This included all the columns except for the target variable (Test Results).
- y (target): This column was the Test Results column, representing the outcome or labels.

Splitting the Data: 
The dataset was split into training and test sets using train_test_split(), ensuring that 20% of the data was used for testing and 80% for training.

Feature Selection (SelectKBest)
Selecting the Best Features: SelectKBest is a feature selection technique that selects the most relevant features for the model by using statistical tests. In this case, 15 features are selected, meaning only the top 15 most informative features are used for model training. This helps reduce dimensionality and computational complexity while improving model performance by eliminating irrelevant features.

Data Scaling:
StandardScaler: The features were scaled using StandardScaler from sklearn.preprocessing, which transforms the data to have a mean of 0 and a standard deviation of 1. This helps in improving the performance of machine learning models that are sensitive to the scale of the data (e.g., SVM, logistic regression).

Model Training:
Model Definition: A variety of models were defined to evaluate performance:
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
Support Vector Classification
Training and Evaluation: Each model was trained using the scaled training data (x_train_scaled and y_train) and evaluated using the test data (x_test_scaled and y_test).
Performance metrics like accuracy, precision, recall, F1-score, and confusion matrix were used to evaluate the models.
Random Forest showed the best performance in terms of accuracy (0.42).

Hyperparameter Tuning:
RandomizedSearchCV was used to randomly sample combinations of hyperparameters and perform cross-validation to select the best configuration. The best parameters were found, and a new Random Forest model was trained using these parameters.
The best parameters were: n_estimators=50, max_depth=30, min_samples_split=2, min_samples_leaf=2, and bootstrap=False.
The tuned model was then retrained and evaluated.

Pipeline Creation and Evaluation:
Pipeline Construction: A pipeline was created using Pipeline() from sklearn to streamline the process of scaling and modeling.
The pipeline first applied StandardScaler() to scale the data, followed by a RandomForestClassifier() for classification.
The pipeline was trained and evaluated with the same performance metrics. The accuracy was slightly higher (0.43) compared to individual models.

Model Saving and Loading:
Saving the Model: The trained pipeline was saved to a .pkl file using joblib.dump() for future use, so it could be reloaded and used to make predictions without retraining.
Loading the Model: The saved pipeline was later reloaded using joblib.load() to evaluate predictions on unseen data. The accuracy remained consistent at 0.7.

Unseen Data Evaluation:
Sampling Unseen Data: 10 random rows were selected from the dataset using data.sample() to test the model's performance on unseen data.
Prediction: The trained model (or pipeline) was used to predict the target values for the unseen data.
Evaluation: Accuracy, confusion matrix, and classification report were computed for the unseen data, showing an accuracy of 0.7, with a comparison of actual vs predicted values.

Saving the Comparison Results:
Comparison of Actual vs Predicted: A DataFrame was created that combined the features of the unseen data along with the actual and predicted target values.
Saving the Comparison: The comparison was saved to a CSV file (unseen_data_comparison.csv) for review.

Evaluation on Unseen Data
The model's performance on unseen data showed an accuracy of 0.7, which is fairly good for a model tested on such a small sample. A confusion matrix was generated, showing how well the predictions matched the actual values. The performance on unseen data seems acceptable given the small sample size, but would need further testing with a larger, more representative dataset.

Conclusion:
The entire process included essential steps for building, evaluating, and tuning a machine learning model. Data preprocessing, feature scaling, model selection, hyperparameter tuning, and evaluation were systematically implemented. The saved pipeline allows future predictions, ensuring the model can be reused without retraining. The comparison of actual vs predicted values also provides insights into the model's prediction quality on unseen data.
