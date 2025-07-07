# Project:3 Trading Signal Prediction Using Technical Indicators and Machine Learning

## Overview

This project explores the application of traditional technical analysis indicators combined with machine learning models to predict stock trading signals — Buy, Sell, or Hold.

The core idea is to manually compute widely used indicators, **MACD** and **RSI**, from raw stock price data, generate trading signals based on these indicators, and then use these signals as ground-truth labels for training classification models. The project implements and compares three machine learning models: Logistic Regression, Random Forest, and Support Vector Machines (SVM), including hyperparameter optimization and evaluation.

---

## Features

- **Manual Computation of Technical Indicators**: MACD and RSI implemented from scratch using Python and Pandas, without relying on external technical analysis libraries.
- **Signal Generation**: Buy/Sell/Hold signals derived from indicator crossovers and thresholds.
- **Data Preparation**: Cleaning, feature integration, and train-validation-test splitting.
- **Model Building and Evaluation**: Implementation and comparison of Logistic Regression, Random Forest, and SVM classifiers.
- **Hyperparameter Tuning**: Grid Search with cross-validation to optimize model performance.
- **Performance Analysis**: Precision, recall, F1-score, confusion matrix, and discussion of results.
  
---

## Getting Started

### Prerequisites

- Python 3.7+
- Pandas, NumPy, scikit-learn, matplotlib (for optional plotting)
- Jupyter Notebook (recommended for running and exploring the code)
- Google Colab or local Jupyter environment

### Installation

Clone the repository:

```bash
git clone https://github.com/Samuel-Solomon-1/Project-3-Machine-Learning-for-Predicting-Trading-Signals/.git
cd Project-3-Machine-Learning-for-Predicting-Trading-Signals
````

Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

1. Prepare your raw stock price CSV dataset.
2. Run the notebook `01_feature_engineering.ipynb` to compute MACD, RSI, and generate trading signals.
3. Proceed with `02_data_preparation.ipynb` to prepare datasets for modeling.
4. Run `03_model_building.ipynb` to train and validate the machine learning models.
5. Use `04_model_optimization.ipynb` to perform hyperparameter tuning and evaluate final model performance.
6. Review insights and conclusions in the final section of the notebooks.

---

## Project Structure

```
├── Project 3: Machine Learning for Predicting Trading Signals.ipynb
├── README.md
├── requirements.txt
```

---

## Results Summary

* Random Forest achieved the best balanced performance after tuning.
* Logistic Regression and SVM struggled with class imbalance.
* Highlighted limitations of relying solely on MACD and RSI.
* Provided recommendations for feature engineering and advanced modeling techniques.

---

## Future Work

* Incorporate additional technical indicators and market data.
* Explore advanced deep learning models (LSTM, transformers) for sequence prediction.
* Use techniques to address class imbalance (SMOTE, focal loss).
* Integrate financial performance metrics (e.g., profit/loss simulation) to evaluate practical trading strategies.
* 
---

## Acknowledgments

* Inspired by classical technical analysis concepts.
* Utilized open-source Python libraries like Pandas and scikit-learn.
* Thanks to the open financial datasets community for accessible stock data.

---

Feel free to contribute, raise issues, or suggest improvements!

---

*Author: Samuel Solomon*
