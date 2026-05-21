from sklearn.metrics import mean_squared_error
import numpy as np


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using RMSE.
    """

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return rmse
