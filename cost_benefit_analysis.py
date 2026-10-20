import price_prediction as pp


def get_feature_coefficients():
    feature_coefficients = {}

    if pp.X_columns is not None and pp.model is not None and pp.scaler is not None:
        for col, coef, scale in zip(
            pp.X_columns, pp.model.coef_, pp.scaler.scale_
        ):
            feature_coefficients[col] = round(float(coef / scale), 2)

    return feature_coefficients