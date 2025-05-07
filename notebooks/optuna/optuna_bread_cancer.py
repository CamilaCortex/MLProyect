# Importamos las librerías necesarias
import numpy as np
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import catboost as cb


# Definimos la función objetivo donde se realiza el entrenamiento y evaluación del modelo
def objective(trial):
    # Cargamos el conjunto de datos de cáncer de mama y separamos características y etiquetas
    data, target = load_breast_cancer(return_X_y=True)

    # Dividimos el conjunto de datos en entrenamiento y validación (70% entrenamiento, 30% validación)
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.3)

    # Definimos un diccionario de parámetros para el modelo
    param = {
        # Sugerimos el tipo de objetivo (función de pérdida) a utilizar
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        # Sugerimos un valor para la proporción de características a usar en cada nivel del árbol
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        # Sugerimos la profundidad máxima del árbol
        "depth": trial.suggest_int("depth", 1, 12),
        # Sugerimos el tipo de boosting (método de construcción de árboles)
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        # Sugerimos el tipo de muestreo que se usará (Bootstrap)
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        # Establecemos un límite en el uso de RAM
        "used_ram_limit": "3gb",
    }

    # Ajustamos parámetros adicionales según el tipo de bootstrap seleccionado
    if param["bootstrap_type"] == "Bayesian":
        # Si el tipo es Bayesiano, sugerimos un valor para la temperatura de muestreo
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        # Si el tipo es Bernoulli, sugerimos un valor para la proporción de datos a muestrear
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    # Creamos una instancia del clasificador CatBoost con los parámetros sugeridos
    gbm = cb.CatBoostClassifier(**param)

    # Entrenamos el modelo usando el conjunto de entrenamiento
    gbm.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=0, early_stopping_rounds=100)

    # Realizamos predicciones sobre el conjunto de validación
    preds = gbm.predict(valid_x)

    # Redondeamos las predicciones a la clase más cercana (0 o 1)
    pred_labels = np.rint(preds)

    # Calculamos la precisión de las predicciones
    accuracy = accuracy_score(valid_y, pred_labels)

    # Retornamos la precisión, que se usará para evaluar la calidad del modelo
    return accuracy

# Función principal
if __name__ == "__main__":
    # Creamos un estudio de Optuna para optimizar los hiperparámetros
    study = optuna.create_study(direction="maximize")  # Queremos maximizar la precisión
    # Iniciamos el proceso de optimización, ejecutando la función objetivo para múltiples ensayos
    study.optimize(objective, n_trials=100, timeout=600)

    # Mostramos el número de ensayos completados
    print("Number of finished trials: {}".format(len(study.trials)))

    # Mostramos el mejor ensayo obtenido
    print("Best trial:")
    trial = study.best_trial

    # Mostramos el valor (precisión) del mejor ensayo
    print("  Value: {}".format(trial.value))

    # Mostramos los parámetros del mejor ensayo
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))