import pandas as pd
from pydantic import BaseModel

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


model = None
scaler = None
X_columns = None
feature_means = None


class CarFeatures(BaseModel):
    wheelbase: float = 98.4
    carlength: float = 176.9
    carwidth: float = 65.4
    carheight: float = 54.3
    curbweight: float = 2335
    enginesize: float = 109
    boreratio: float = 3.19
    stroke: float = 3.40
    compressionratio: float = 10.0
    horsepower: float = 102
    peakrpm: float = 5500
    citympg: float = 24
    highwaympg: float = 30
    CarName: str = "toyota"
    fueltype: str = "gas"
    aspiration: str = "std"
    doornumber: str = "four"
    carbody: str = "sedan"
    drivewheel: str = "fwd"
    enginelocation: str = "front"
    enginetype: str = "ohc"
    cylindernumber: str = "four"
    fuelsystem: str = "mpfi"


def train_model():
    global model, scaler, X_columns, feature_means

    car = pd.read_csv("CarPrice_Assignment.csv")

    car.drop(["car_ID","symboling"], axis=1, inplace=True)
    car["CarName"] = car["CarName"].map(lambda x: str(x).split()[0])
    car["CarName"] = car["CarName"].replace(
        {
            "maxda": "mazda",
            "Nissan": "nissan",
            "porcshce": "porsche",
            "toyouta": "toyota",
            "vokswagen": "volkswagen",
            "vw": "volkswagen"
        }
    )

    cat_columns = car.select_dtypes(include="object").columns
    car = pd.get_dummies(car, columns=cat_columns, drop_first=True, dtype=int)

    y = car["price"].values
    X = car.drop("price", axis=1)
    X_columns = X.columns
    feature_means = X.mean().to_dict()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)


def predict(car: CarFeatures):
    input_dict = car.dict()
    input_df = pd.DataFrame([input_dict])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)

    predicted_price = float(model.predict(input_scaled)[0])

    brand_col = f"CarName_{car.CarName.lower()}"
    brand_impact = 0.0
    if brand_col in X_columns:
        idx = list(X_columns).index(brand_col)
        brand_impact = float(
            model.coef_[idx]
            * (1 - scaler.mean_[idx])
            / scaler.scale_[idx]
        )

    engine_idx = list(X_columns).index("enginesize")
    engine_impact = float(
        model.coef_[engine_idx]
        * (car.enginesize - scaler.mean_[engine_idx])
        / scaler.scale_[engine_idx]
    )

    hp_idx = list(X_columns).index("horsepower")
    hp_impact = float(
        model.coef_[hp_idx]
        * (car.horsepower - scaler.mean_[hp_idx])
        / scaler.scale_[hp_idx]
    )

    return {
        "predicted_price": round(predicted_price, 2),
        "confidence_lower": round(predicted_price * 0.88, 2),
        "confidence_upper": round(predicted_price * 1.12, 2),
        "brand_impact": round(brand_impact, 2),
        "engine_impact": round(engine_impact, 2),
        "hp_impact": round(hp_impact, 2),
    }
