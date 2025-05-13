import logging
from predict_test import prepare_features, predict  # Asegúrate de que estas funciones están definidas

# Configura el logging para mostrar mensajes en la consola
logging.basicConfig(level=logging.INFO)

# Datos de entrada para la predicción
ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

if __name__ == "__main__":
    features = prepare_features(ride)
    pred = predict(features)
    logging.info(f"Prediction: {pred}")
    print(f"Prediction: {pred}")