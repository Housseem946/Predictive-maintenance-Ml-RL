# Predictive Maintenance System for Industrial Equipment

A predictive maintenance system using machine learning and reinforcement learning to forecast equipment failures and optimize maintenance schedules, leveraging the NASA CMAPSS dataset.

## Overview

This repository contains the implementation of a predictive maintenance system for industrial equipment using machine learning and reinforcement learning techniques. The project integrates data preprocessing, ensemble learning, AutoML, and reinforcement learning to build a scalable and efficient solution. The NASA CMAPSS dataset is used to predict equipment failures and optimize maintenance schedules.

## Project Goals

1. **Predict Equipment Failures:** Develop a machine learning model to predict equipment failures based on sensor data.
2. **Handle Imbalanced Data:** Address challenges with imbalanced datasets where failures are rare.
3. **Optimize Maintenance Scheduling:** Use reinforcement learning to optimize maintenance policies.
4. **Automate Model Selection:** Leverage AutoML to streamline model selection and hyperparameter tuning.

## Key Deliverables

- **Jupyter Notebooks:**
  - Data Preprocessing
  - Model Training and Evaluation
  - Reinforcement Learning Implementation
- **AutoML Pipeline:** TPOT-exported pipeline for automated model selection.
- **Final Report:** Detailed analysis, results, and recommendations for future work.

## Project Structure
```
├── notebooks
│   ├── preprocessing.ipynb        # Data preprocessing and feature engineering
│   ├── model_training.ipynb       # Model training and evaluation
│   ├── reinforcement_learning.ipynb  # Q-Learning implementation for maintenance scheduling
├── tpot_best_pipeline.py          # Exported pipeline from TPOT AutoML
├── data
│   ├── train_FD001.txt            # Training dataset
│   ├── test_FD001.txt             # Testing dataset
│   ├── RUL_FD001.txt              # Remaining Useful Life (RUL) data
├── reports
│   ├── final_report.pdf           # Comprehensive project report
├── README.md                      # Project documentation
```

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/predictive-maintenance-system.git](https://github.com/Housseem946/Predictive-maintenance-Ml-RL.git)
   cd predictive-maintenance-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run Jupyter notebooks to reproduce the results:
   ```bash
   jupyter notebook
   ```

## Key Features

- **Ensemble Learning:**
  - Random Forest and XGBoost models for robust failure predictions.
- **AutoML:**
  - TPOT for automated model selection and hyperparameter optimization.
- **Reinforcement Learning:**
  - Q-Learning algorithm for optimizing maintenance schedules.
- **Imbalanced Data Handling:**
  - Techniques like SMOTE and cost-sensitive learning.

## Results Summary

| Model            | Accuracy | Precision | Recall | AUC-ROC |
|------------------|----------|-----------|--------|---------|
| Random Forest    | 88.0%    | 87.0%     | 85.0%  | 0.90    |
| XGBoost          | 91.0%    | 90.0%     | 89.0%  | 0.92    |
| TPOT AutoML      | 92.0%    | 93.0%     | 92.0%  | 0.94    |


## Acknowledgments

- **NASA CMAPSS Dataset:** Used for training and testing the models.
- **TPOT Library:** For AutoML pipeline generation.
- **Scikit-learn & Imbalanced-learn Libraries:** For machine learning model training and data handling.
- **Matplotlib & Seaborn:** For visualizations.

## Author 
-  Houssem

For any questions, you can contact me via LinkedIn : https://www.linkedin.com/in/houssem-rezgui-/

## Contributions
Contributions are welcome! Please open an issue or submit a pull request for enhancements.
