# CO2 Emissions Prediction Model

This Colab notebook builds and trains a model to predict CO2 emissions of vehicles in Canada based on various features.

## Description

The notebook uses a dataset of CO2 emissions for various vehicle models in Canada. It performs data preprocessing, feature engineering, and model training using a RandomForestRegressor. The model is then evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) score.

## Usage

1. **Data Loading:** The notebook loads the CO2 emissions dataset from a CSV file.
2. **Data Preprocessing:** Irrelevant columns are dropped, and the data is split into training and testing sets.
3. **Model Training:** A pipeline is created using ColumnTransformer for feature scaling and encoding. A RandomForestRegressor is used for prediction.
4. **Hyperparameter Tuning:** GridSearchCV is used to find the best hyperparameters for the model.
5. **Model Evaluation:** The model is evaluated using various metrics.
6. **Model Saving:** The trained model is saved using joblib for later use.
7. **Cargo Vehicle Prediction:** A function is defined to predict emissions for cargo vehicles considering their weight.
8. **Example Prediction:** An example prediction is shown for a pickup truck with a specified weight.

## Dependencies

The notebook requires the following libraries:

- numpy
- pandas
- scikit-learn
- joblib

You can install them using `pip install numpy pandas scikit-learn joblib`.

## How to Run

1. Open the notebook in Google Colab.
2. Run all the cells in sequence.
3. You can modify the input parameters in the "Example Prediction" section to predict emissions for different vehicles.

## Results

The model achieves a reasonable accuracy in predicting CO2 emissions. To see the results and evaluation metrics, run the code.

## Notes

- The model is trained on a specific dataset and may not generalize well to other datasets.
- The cargo vehicle prediction function uses a simplified approach to adjust emissions linearly,based on weight.
- Further improvements can be made by exploring other models and feature engineering techniques.
