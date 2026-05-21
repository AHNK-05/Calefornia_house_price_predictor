from sklearn.ensemble import RandomForestRegressor


def train_model(X_train, y_train):
    """
    Train Random Forest model.
    """

    model = RandomForestRegressor()

    model.fit(X_train, y_train)

    return model
