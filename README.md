# Diabetes Data Cleanup Project

# What This Project Does

This project is the cleanup and preparation phase for building a machine learning model to predict diabetes.

The Goal: Take messy, real-world health data (like patient records with errors and missing numbers) and make it perfect for a computer to learn from.

We use a special Python program (data_prep_pipeline.py) to run through five main cleanup steps, or "Phases."

---

##  Project Files Explained

| Folder/File | Purpose |
| :--- | :--- |
| data/Diabetes Missing Data.csv | INPUT: The original, raw dataset we start with. |
| data/final_processed_diabetes_data.csv | OUTPUT: The final, clean, ready-to-use dataset (saved after running the notebook). |
| src/data_prep_pipeline.py | The main reusable Python code that does all the cleaning work. |
| notebook/diabetes_data_prep.ipynb | The step-by-step guide (Jupyter Notebook) you run to see the process and all the results. |
| README.md | This file! |

---

##  How to Set Up (Install the Tools)

You need to install some key Python tools to run this project.

1.  Open your terminal/command line.
2.  Install everything you need using one of the commands below:

| If your computer recognizes... | Use this command: |
| :--- | :--- |
| pip | pip install pandas numpy scikit-learn matplotlib seaborn missingno imbalanced-learn tabulate jupyter |
| py | py -m pip install pandas numpy scikit-learn matplotlib seaborn missingno imbalanced-learn tabulate jupyter |

---

##  The 5 Steps of Data Preparation

The diabetes_data_prep.ipynb notebook takes you through these steps:

| Step | What We Fix | Why It Matters |
| :--- | :--- | :--- |
| 1. Understand the Data | Find obvious problems, like zeros in Glucose or BMI (which aren't possible). | We need to know what's broken before we fix it. |
| 2. Clean the Data | Replace bad zeros with "missing" (NaN), then fill those missing spots with the median (the middle value). We also smooth out extreme outliers. | Fills in the blanks so the computer can use all the data. |
| 3. Transform the Data | Turn numerical values (like a patient's exact age) into easy groups (like "Young" or "Senior"). Then, we scale all numbers. | Makes complex numbers simpler and prevents big numbers from overpowering small ones. |
| 4. Reduce the Data | Find the most important features (like Glucose and BMI) and remove the less useful ones. | Speeds up the model and prevents confusion (multicollinearity). |
| 5. Balance the Data | Fix the problem where we have many non-diabetic examples (0) but few diabetic examples (1). We use SMOTE to create synthetic diabetic examples. | Ensures the model learns how to predict the rare "diabetic" case accurately. |

---

##  How to Run It

1.  Make sure the CSV data file is in the data folder.
2.  In your terminal, run: jupyter notebook
3.  Click on notebook/diabetes_data_prep.ipynb and run each cell in order.
4. if you see the error first all file save and restart kernel then run all cell 

The notebook will show you pictures and reports at every step!
