# Flight Fare Price Prediction

## Project Overview

This project aims to predict flight ticket prices based on various features such as airline, source, destination, duration, number of stops, and travel dates. The dataset used is sourced from a Kaggle dataset, and the project employs machine learning techniques to build a predictive model. A user-friendly web application is also developed using Streamlit to allow users to input flight details and receive price predictions.

## Dataset

The dataset (`Data_Train.xlsx`) contains the following features:

- **Airline**: Name of the airline (e.g., IndiGo, Air India, Jet Airways).
- **Date_of_Journey**: Date of the flight.
- **Source**: Departure city (e.g., Bangalore, Kolkata, Delhi).
- **Destination**: Arrival city (e.g., New Delhi, Bangalore, Cochin).
- **Route**: Flight route with intermediate stops.
- **Dep_Time**: Departure time.
- **Arrival_Time**: Arrival time.
- **Duration**: Total flight duration.
- **Total_Stops**: Number of stops (non-stop, 1 stop, 2 stops, etc.).
- **Additional_Info**: Extra information (e.g., "No info", "In-flight meal not included").
- **Price**: Target variable (flight ticket price in INR).

The dataset is preprocessed to handle missing values, duplicates, and categorical variables, and features are engineered for model training.

## Project Structure

The project is implemented in a Jupyter Notebook (`fare-price-prediction.ipynb`) and includes the following sections:

1. **Data Loading and Exploration**:

   - Libraries: `pandas`, `plotly.express`, `plotly.io`.
   - Initial data inspection using `df.head()`, `df.tail()`, `df.info()`, and `df.describe()`.
   - Checking for null values (`df.isna().sum()`) and duplicates (`df.duplicated().sum()`).

2. **Exploratory Data Analysis (EDA)**:

   - Univariate analysis of categorical features like `Airline` using `value_counts()`.
   - Visualization of airline distribution using `plotly.express`.

3. **Data Preprocessing**:

   - Handling missing values by dropping rows with nulls (`df.dropna()`).
   - Removing duplicates (`df.drop_duplicates()`).
   - Feature engineering:
     - Converting `Date_of_Journey` to `Day` and `Month`.
     - Parsing `Dep_Time` and `Arrival_Time` into hours and minutes.
     - Converting `Duration` to minutes.
     - Encoding `Total_Stops` numerically (e.g., "non-stop" to 0, "1 stop" to 1).
   - Dropping unnecessary columns like `Route`.

4. **Model Training**:

   - **Pipeline**:
     - Numerical features (`Duration`, `Total_Stops`, `Day`, `Month`, `Dep_Hour`, `Dep_Min`, `Arrival_Hour`, `Arrival_Min`) are scaled using `StandardScaler`.
     - Categorical features (`Airline`, `Source`, `Destination`, `Additional_Info`) are encoded using `OneHotEncoder`.
   - **Model**: `GradientBoostingRegressor` is used due to its low error rate.
   - **Hyperparameter Tuning**: `GridSearchCV` is employed to optimize parameters (`n_estimators`, `learning_rate`, `max_depth`, `max_leaf_nodes`).
   - **Evaluation**: The model is evaluated using `neg_root_mean_squared_error`.

5. **Model Saving**:

   - The trained model is saved as `price_prediction.pkl`.
   - Input column names are saved as `inputs.pkl` for the web app.

6. **Web Application**:

   - A Streamlit app (`app.py`) is created for user interaction.
   - Users can input flight details via sliders and dropdowns, and the app predicts the fare using the saved model.

## Dependencies

The project requires the following Python libraries:

- `pandas`
- `plotly.express`
- `plotly.io`
- `scikit-learn`
- `joblib`
- `streamlit`

Install dependencies using:

```bash
pip install pandas plotly scikit-learn joblib streamlit
```

## Results

- **Best Model Parameters** (from `GridSearchCV`):
  - `learning_rate`: 0.1
  - `max_depth`: 10
  - `max_leaf_nodes`: 20
  - `n_estimators`: 300
- **Performance**:
  - Mean Train RMSE: \~792.74
  - Mean Test RMSE: \~1094.67
- The `GradientBoostingRegressor` was selected for its superior performance in minimizing prediction errors.


## Acknowledgements

- Dataset sourced from Kaggle.
- Built with Python, scikit-learn, and Streamlit.