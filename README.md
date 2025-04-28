# Movie Rating Prediction

This project aims to predict IMDb movie ratings based on various features like Genre, Director, Actors, Year, Duration, and Votes. It involves data cleaning, preprocessing, feature engineering, model training (using RandomForestRegressor), and provides an interactive interface for predicting the rating of a new movie.

**Project Date:** April 28, 2025 (as mentioned in the document)

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Features](#features)
3.  [Dataset](#dataset)
4.  [Workflow](#workflow)
5.  [Installation](#installation)
6.  [Usage](#usage)
7.  [Evaluation](#evaluation)
8.  [Future Improvements](#future-improvements)

## Project Overview

The goal is to build a predictive model that estimates a movie's IMDb rating. The process includes loading the dataset, performing extensive cleaning and preprocessing to handle missing values and inconsistent formats, engineering new features like average ratings for directors and genres, training a machine learning model, and evaluating its performance. Finally, an interactive command-line tool allows users to input movie details and get a predicted rating.

## Features

*   **Data Loading and Cleaning:** Reads data from a CSV file, handles encoding issues, cleans numerical columns (Year, Duration, Votes), and identifies missing values.
*   **Preprocessing:**
    *   Handles missing 'Rating' values by dropping corresponding rows.
    *   Imputes missing numerical features (Year, Duration, Votes) using the median strategy.
    *   Imputes missing categorical features (Genre, Director, Actors) with the placeholder 'Unknown'.
*   **Feature Engineering:**
    *   Extracts the primary genre from the 'Genre' column.
    *   Calculates the average rating for each Director based on the *training data*.
    *   Calculates the average rating for each Primary Genre based on the *training data*.
*   **Encoding & Scaling:**
    *   Applies One-Hot Encoding to the 'Primary_Genre' column.
    *   Applies StandardScaler to numerical features (including engineered average ratings).
*   **Model Training:** Trains a `RandomForestRegressor` model on the processed features.
*   **Evaluation:** Evaluates the model performance on the test set using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²) metrics.
*   **Interactive Prediction:** Provides a command-line interface to input details for a new movie (Year, Duration, Votes, Director, Primary Genre) and predicts its rating using the trained model and preprocessing pipeline.

## Dataset

The dataset used is `movie project data.csv` (path suggests local download). It contains information about movies with the following key columns:

*   `Name`: Title of the movie (Dropped before training).
*   `Year`: Release year.
*   `Duration`: Runtime in minutes.
*   `Genre`: Genre(s) of the movie.
*   `Rating`: IMDb rating (Target Variable).
*   `Votes`: Number of votes received.
*   `Director`: Director's name.
*   `Actor 1`, `Actor 2`, `Actor 3`: Names of key actors.

The dataset initially contains a significant number of missing values across multiple columns.

## Workflow

1.  **Load Data:** Load the dataset using pandas, specifying 'cp1252' encoding.
2.  **Initial EDA & Cleaning:** Explore data shape, head, info, and initial missing values. Drop rows where the target variable 'Rating' is missing.
3.  **Clean Specific Columns:** Convert 'Year', 'Duration', and 'Votes' to numerical types, handling formatting inconsistencies (e.g., removing ' min', ',').
4.  **Impute Missing Values:**
    *   Use `SimpleImputer` (median) for numerical columns ('Year', 'Duration', 'Votes').
    *   Fill NaN in categorical columns ('Genre', 'Director', 'Actor 1/2/3') with 'Unknown'.
5.  **Feature Engineering:**
    *   Create 'Primary_Genre' from the 'Genre' column.
    *   Calculate `Director_Avg_Rating` and `Genre_Avg_Rating` based *only* on the training set to prevent data leakage. Fill any NaNs in these new features (e.g., for directors/genres not seen in training) with the overall mean rating of the training set.
6.  **Train-Test Split:** Split the data into training (80%) and testing (20%) sets.
7.  **Encoding & Scaling:**
    *   Apply `OneHotEncoder` to 'Primary_Genre' (fitting only on training data).
    *   Apply `StandardScaler` to all numerical features (fitting only on training data).
    *   Ensure train and test sets have the same columns after encoding, handling potential discrepancies.
8.  **Drop Original Columns:** Remove original 'Genre', 'Name', 'Director', 'Actor 1/2/3', and 'Primary_Genre' columns after processing/encoding.
9.  **Model Training:** Train a `RandomForestRegressor` (n_estimators=100, max_depth=15, min_samples_leaf=5, oob_score=True, random_state=42) on the prepared training data.
10. **Model Evaluation:** Predict ratings on the test set and calculate MAE, RMSE, and R².
11. **Interactive Prediction:** Set up helper functions and a prediction pipeline function (`predict_rating`) that encapsulates all preprocessing steps using the fitted objects (imputer, scaler, OHE, average maps) to predict the rating for user-provided movie details. Run the interactive loop.

## Installation

1.  Clone the repository or download the source files.
2.  Ensure you have Python 3.x installed.
3.  Install the required libraries. It's recommended to use a virtual environment.
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn notebook
    ```
    Alternatively, if a `requirements.txt` file is provided:
    ```bash
    pip install -r requirements.txt
    ```
4.  Place the `movie project data.csv` file in the expected location or update the path in the script/notebook.

## Usage

1.  Open the Jupyter Notebook (`.ipynb` file) or run the Python script (`.py` file).
    *   **Jupyter Notebook:** Run all cells sequentially (`Kernel -> Restart & Run All`).
    *   **Python Script:** Execute the script from your terminal: `python your_script_name.py`
2.  The script will perform data loading, cleaning, preprocessing, training, and evaluation.
3.  After the model evaluation, the interactive prediction section will start.
4.  Follow the prompts to enter the movie details:
    *   **Year:** Enter the release year (numeric).
    *   **Duration:** Enter the duration in minutes (numeric).
    *   **Votes:** Enter the approximate number of votes (numeric).
    *   **Director:** Select a director from the top 100 list or choose 'Other' to enter manually.
    *   **Primary Genre:** Select the main genre from the provided list.
5.  Press Enter at numerical prompts to use default values (medians from the training data).
6.  The script will output the predicted IMDb rating based on your inputs.
7.  You will be prompted whether to predict another movie or exit.

## Evaluation

The `RandomForestRegressor` model was evaluated on the test set with the following results:

*   **Mean Absolute Error (MAE):** ~0.9175
*   **Root Mean Squared Error (RMSE):** ~1.2211
*   **R-squared (R²):** ~0.1979

*Note: The R² score is relatively low, suggesting that the current features and model explain only about 19.8% of the variance in movie ratings in the test set. The model's predictive power might be limited.*

## Future Improvements

*   **Advanced Feature Engineering:** Explore interactions between features (e.g., Director-Genre), use actor combinations, or incorporate NLP on movie names/synopses if available.
*   **Hyperparameter Tuning:** Use techniques like GridSearchCV or RandomizedSearchCV to find optimal parameters for the RandomForestRegressor or other models.
*   **Try Different Models:** Experiment with other regression algorithms like Gradient Boosting (XGBoost, LightGBM), Support Vector Regression, or even simple Neural Networks.
*   **Handle 'Unknown' Category:** Explore more sophisticated ways to handle missing categorical data instead of just using 'Unknown'.
*   **More Data:** Acquire a larger or more feature-rich dataset if possible.
*   **Cross-Validation:** Implement k-fold cross-validation during training for more robust evaluation.
