# src/data_prep_pipeline.py

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPrepPipeline:
    """
    A class to handle data loading, initial exploration, cleaning, and 
    transformation for the Diabetes Prediction project.
    """
    def __init__(self, data_path):
        """Initializes the pipeline with the data file path."""
        self.data_path = data_path
        self.df = None
        self.quality_issues = {}

    def load_data(self):
        """Task 1: Load the diabetes dataset."""
        print(f"Loading data from: {self.data_path}")
        try:
            self.df = pd.read_csv(self.data_path)
            print("Data loaded successfully.")
            return self.df
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_path}")
            self.df = None
            return None

    def initial_exploration(self):
        """Task 2 & 3: Perform initial exploration and identify quality issues."""
        if self.df is None:
            print("Data not loaded. Please run load_data() first.")
            return None

        print("\n--- Initial Data Exploration ---")
        print("Shape:", self.df.shape)
        print("\nData Types:")
        print(self.df.dtypes)
        print("\nFirst 5 Rows:")
        print(self.df.head())
        print("\nSummary Statistics:")
        print(self.df.describe().T)
        print("\nMissing Values (NaNs):")
        print(self.df.isnull().sum())

        self.suspect_features = ['Glucose', 'Diastolic_BP', 'Skin_Fold', 
                                 'Serum_Insulin', 'BMI']
        
        print("\n--- Data Quality Issues (Zeros) ---")
        for feature in self.suspect_features:
            zero_count = (self.df[feature] == 0).sum()
            percent_zero = (zero_count / len(self.df)) * 100
            self.quality_issues[feature] = {
                'zero_count': zero_count,
                'percent_zero': f"{percent_zero:.2f}%"
            }
            if zero_count > 0:
                print(f"Feature **{feature}** has **{zero_count}** ( {percent_zero:.2f}% ) zero values (biologically impossible/suspect).")

        return self.quality_issues

    def missing_value_analysis(self):
        """Phase 2, Task 1: Identify zeros in suspect features and replace with NaN."""
        if self.df is None:
            print("Data not loaded.")
            return

        df_copy = self.df.copy()

        print("\n--- Missing Value Analysis ---")
        self.missing_data_percentages = {}

        for feature in self.suspect_features:
            df_copy[feature] = df_copy[feature].replace(0, np.nan)

        missing_counts = df_copy.isnull().sum()
        total_rows = len(df_copy)
        
        for feature in df_copy.columns:
            percent = (missing_counts[feature] / total_rows) * 100
            self.missing_data_percentages[feature] = f"{percent:.2f}%"
            if percent > 0:
                 print(f"**{feature}**: {missing_counts[feature]} missing values ({percent:.2f}%)")
        
        self.df = df_copy 
        return self.missing_data_percentages

    def impute_data(self, strategy='median'):
        """Phase 2, Task 2: Apply imputation strategy."""
        if self.df is None:
            print("Data not loaded.")
            return

        print(f"\n--- Imputation Strategy: **{strategy.upper()}** ---")
        
        for feature in self.suspect_features:
            if self.df[feature].isnull().any():
                fill_value = self.df[feature].median() if strategy == 'median' else self.df[feature].mean()
                self.df[feature].fillna(fill_value, inplace=True)
                print(f"Imputed **{feature}** with the **{strategy}** value: {fill_value:.2f}")

        print("\nMissing values after imputation:")
        print(self.df.isnull().sum())
    
    def outlier_treatment(self, method='IQR', cap_method='capping'):
        """Phase 2, Task 3: Outlier Detection & Treatment using IQR."""
        if self.df is None:
            print("Data not loaded.")
            return
        
        print(f"\n--- Outlier Treatment: **{method.upper()}** (using **{cap_method}**) ---")
        
        numerical_features = self.df.drop(columns=['Pregnant', 'Class']).columns 

        for feature in numerical_features:
            Q1 = self.df[feature].quantile(0.25)
            Q3 = self.df[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_count = ((self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)).sum()
            
            if outliers_count > 0:
                print(f"**{feature}** has **{outliers_count}** outliers (IQR method).")
                
                if cap_method == 'capping':
                    self.df[feature] = np.where(self.df[feature] > upper_bound, upper_bound, self.df[feature])
                    self.df[feature] = np.where(self.df[feature] < lower_bound, lower_bound, self.df[feature])
                    print(f"  -> Outliers capped between **{lower_bound:.2f}** and **{upper_bound:.2f}**.")
                

    def feature_engineering(self):
        """Phase 3, Task 1: Create categorical features."""
        if self.df is None:
            print("Data not loaded.")
            return

        print("\n--- Feature Engineering ---")

        # 1. Create Age Groups (Ordinal Categorical)
        bins_age = [0, 25, 40, 60, self.df['Age'].max() + 1]
        labels_age = ['Young', 'Middle-Aged', 'Senior', 'Elderly']
        self.df['Age_Group'] = pd.cut(self.df['Age'], bins=bins_age, labels=labels_age, right=False)
        print("-> Created **Age_Group** feature.")

        # 2. Calculate BMI Categories (Ordinal Categorical)
        bins_bmi = [0, 18.5, 25, 30, self.df['BMI'].max() + 1]
        labels_bmi = ['Underweight', 'Normal', 'Overweight', 'Obese']
        self.df['BMI_Category'] = pd.cut(self.df['BMI'], bins=bins_bmi, labels=labels_bmi, right=False)
        print("-> Created **BMI_Category** feature.")

        def categorize_glucose(g):
            if g < 140:
                return 'Normal_Glucose'
            elif g < 200:
                return 'Pre_Diabetic'
            else:
                return 'Diabetic_Glucose'
                
        self.df['Glucose_Category'] = self.df['Glucose'].apply(categorize_glucose)
        print("-> Created **Glucose_Category** feature.")
        
        self.df.drop(columns=['Age', 'BMI', 'Glucose'], inplace=True)
        print("\nDropped original 'Age', 'BMI', and 'Glucose' features.")
        
    def feature_encoding(self):
        """Phase 3, Task 2: Apply encoding to categorical features."""
        if self.df is None:
            print("Data not loaded.")
            return

        print("\n--- Feature Encoding ---")
        
        age_mapping = {'Young': 0, 'Middle-Aged': 1, 'Senior': 2, 'Elderly': 3}
        self.df['Age_Group_Encoded'] = self.df['Age_Group'].map(age_mapping)
        
        bmi_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
        self.df['BMI_Category_Encoded'] = self.df['BMI_Category'].map(bmi_mapping)
        
        print("-> Applied **Label Encoding** to Age_Group and BMI_Category.")

        glucose_ohe = pd.get_dummies(self.df['Glucose_Category'], prefix='Glucose')
        self.df = pd.concat([self.df, glucose_ohe], axis=1)
        
        print("-> Applied **One-Hot Encoding** to Glucose_Category.")
        
        self.df.drop(columns=['Age_Group', 'BMI_Category', 'Glucose_Category'], inplace=True)


    def feature_scaling(self, scaler_type='StandardScaler'):
        """Phase 3, Task 3: Apply chosen scaling method."""
        if self.df is None:
            print("Data not loaded.")
            return

        print(f"\n--- Feature Scaling (Method: **{scaler_type}**) ---")
        numerical_features = self.df.select_dtypes(include=np.number).columns.tolist()
        features_to_exclude = ['Class', 'Age_Group_Encoded', 'BMI_Category_Encoded']
        features_to_exclude.extend([col for col in numerical_features if col.startswith('Glucose_')]) 
        
        scaling_cols = [col for col in numerical_features if col not in features_to_exclude]
        print(f"Features being scaled: {scaling_cols}")

        if scaler_type == 'StandardScaler':
            scaler = StandardScaler()
        elif scaler_type == 'MinMaxScaler':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler_type. Choose 'StandardScaler' or 'MinMaxScaler'.")

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', scaler, scaling_cols)
            ],
            remainder='passthrough' # Keep other columns (encoded, target) as is
        )

        scaled_data = preprocessor.fit_transform(self.df)
        scaled_col_names = [f'{col}_Scaled' for col in scaling_cols]
        passthrough_cols = [col for col in self.df.columns if col not in scaling_cols]

        self.df = pd.DataFrame(
            scaled_data, 
            columns=scaled_col_names + passthrough_cols
        )
        
        cols = [col for col in self.df.columns if col != 'Class'] + ['Class']
        self.df = self.df[cols]
        
        print(f"-> Applied {scaler_type} to numerical features.")

    def feature_selection(self, k_features=8):
        """Phase 4, Task 1: Analyze correlation and apply SelectKBest."""
        if self.df is None:
            print("Data not loaded.")
            return

        print("\n--- Feature Selection ---")
        X = self.df.drop('Class', axis=1)
        y = self.df['Class'].astype(int)
        
        # 1. Analyze Correlation Matrix
        print("\n**1. Correlation Analysis:**")
        corr_matrix = self.df.corr()
        target_corr = corr_matrix['Class'].sort_values(ascending=False)
        print("Correlation with Target ('Class'):")
        print(target_corr)

        print(f"\n**2. SelectKBest (k={k_features}) with Mutual Information:**")
        
        selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
        selector.fit(X, y)
        
        # Get the scores and the selected features
        scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
        selected_features_mask = selector.get_support()
        selected_features = X.columns[selected_features_mask].tolist()
        
        print(f"Top {k_features} features selected by SelectKBest (scores):")
        print(scores.head(k_features))
        
        self.df = self.df[selected_features + ['Class']]
        print(f"\nDataFrame shape after feature selection: {self.df.shape}")
        
        self.selected_features = selected_features

    def dimensionality_reduction(self):
        """Phase 4, Task 2: Perform PCA to visualize variance explained and determine optimal components."""
        if self.df is None or 'selected_features' not in self.__dict__:
            print("Data not ready. Run feature_selection() first.")
            return

        print("\n--- Dimensionality Reduction (PCA) ---")
        X = self.df.drop('Class', axis=1)
        
        pca = PCA(n_components=X.shape[1])
        pca.fit(X)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        print("Variance Explained by each principal component:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"Component {i+1}: {var:.4f} (Cumulative: {cumulative_variance[i]:.4f})")
            
        optimal_components = np.where(cumulative_variance >= 0.90)[0][0] + 1
        print(f"\nOptimal number of components to explain >90% variance: **{optimal_components}**")
        
        self.pca_variance_ratio = pca.explained_variance_ratio_
        self.pca_cumulative_variance = cumulative_variance
        
        return self.pca_cumulative_variance

    def class_distribution_analysis(self):
        """Phase 5, Task 1: Visualize target class distribution and calculate imbalance ratio."""
        if self.df is None:
            print("Data not loaded.")
            return

        print("\n--- Class Distribution Analysis ---")
        
        class_counts = self.df['Class'].value_counts()
        total_samples = len(self.df)
        
        print("Target Class Counts:")
        print(class_counts)
        
        # Calculate imbalance ratio
        minority_count = class_counts.min()
        majority_count = class_counts.max()
        imbalance_ratio = majority_count / minority_count
        
        print(f"\nImbalance Ratio (Majority / Minority): **{imbalance_ratio:.2f}**")
        print(f"Percentage of Minority Class (1/Diabetic): {(minority_count / total_samples) * 100:.2f}%")
        
        return class_counts

    def handle_imbalance(self, method='SMOTE'):
        """Phase 5, Task 2: Apply appropriate data imbalance handling mechanism."""
        if self.df is None:
            print("Data not loaded.")
            return

        print(f"\n--- Data Imbalance Handling (Method: **{method}**) ---")
        
        X = self.df.drop('Class', axis=1)
        y = self.df['Class'].astype(int) 

        if method == 'SMOTE':
            smote = SMOTE(sampling_strategy='minority', random_state=42) 
            
            # Apply SMOTE
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            print(f"Original samples: {len(self.df)}")
            print(f"Resampled samples: {len(X_resampled)}")

            # Recombine into a single balanced DataFrame
            self.df_processed = pd.DataFrame(X_resampled, columns=X.columns)
            self.df_processed['Class'] = y_resampled
            
            print("\nClass distribution after SMOTE:")
            print(self.df_processed['Class'].value_counts())
            
        else:
            print("Unsupported balancing method.")
            return

        # Update the working DataFrame to the processed one
        self.df = self.df_processed
        print("\nData balancing complete. The final processed dataset is ready.")
        
    def save_final_dataset(self, filename='final_processed_diabetes_data.csv'):
        """Saves the final processed dataset to the data folder."""
        if self.df is None:
            print("Cannot save: Processed data is not available.")
            return
            
        save_path = f'../data/{filename}'
        self.df.to_csv(save_path, index=False)
        print(f"\n**Final Clean Dataset saved to: {save_path}**")
        
        # Create a basic data dictionary
        data_dict = pd.DataFrame({
            'Feature': self.df.columns,
            'Description': [
                "Number of times pregnant (Scaled)", 
                "Plasma glucose concentration (Scaled)", 
                "Diastolic blood pressure (Scaled)",
                "Triceps skin fold thickness (Scaled)", 
                "2-Hour serum insulin (Scaled)",
                "Diabetes pedigree function (Scaled)", 
                "Label-encoded Age Group (0=Young, 3=Elderly)",
                "Label-encoded BMI Category (0=Underweight, 3=Obese)",
                "One-Hot Encoded: Normal Glucose Level",
                "One-Hot Encoded: Pre-Diabetic Glucose Level",
                "One-Hot Encoded: Diabetic Glucose Level",
                "Target variable (0 = Non-diabetic, 1 = Diabetic)"
            ][:len(self.df.columns)]
        })
        print("\nData Dictionary:")
        print(data_dict.to_markdown(index=False))