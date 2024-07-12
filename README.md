# HR Employee Attrition Prediction

## Project Overview

This project aims to predict employee attrition (i.e., whether an employee will leave the company) using a dataset containing various HR-related features. We perform extensive Exploratory Data Analysis (EDA), Principal Component Analysis (PCA), and correlation analysis. We also build several classification models to predict employee attrition.

## Dataset

The dataset used in this project contains information on employees such as:

- Employee satisfaction level
- Last evaluation
- Number of projects
- Average monthly hours
- Time spent at the company
- Whether they have had a work accident
- Whether they have had a promotion in the last 5 years
- Departments
- Salary levels
- Attrition (target variable)

## Project Structure

The project is organized into the following main sections:

1. **Data Exploration and Preprocessing**
    - Load and clean the dataset
    - Handle missing values (if any)
    - Encode categorical variables
    - Feature scaling

2. **Exploratory Data Analysis (EDA)**
    - Visualize the distribution of features
    - Analyze relationships between features and the target variable
    - Create heatmaps to identify correlations
    - Perform PCA to reduce dimensionality and visualize data

3. **Model Building and Evaluation**
    - Split the data into training and testing sets
    - Build and evaluate several classification models:
        - Logistic Regression
        - Random Forest
        - KNN
        - Gradient Boosting
        - Stacking models
    - Compare model performance using metrics such as accuracy, precision, recall, and F1-score

4. **Conclusion and Future Work**
    - Summarize key findings and insights from the analysis
    - Discuss the performance of different models
    - Suggest potential improvements and future work

## Installation

To run this project, you need to have Python installed on your machine along with the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```


## Results

The results of the analysis and model evaluations will be displayed within the Jupyter Notebook or the output of the Python script. Key metrics and visualizations will help you understand the factors influencing employee attrition and the performance of the different classification models.

## Contributing

Contributions to this project are welcome. If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.
