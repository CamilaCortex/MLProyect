import pickle
import logging
from flask import Flask, request, jsonify

with open('lin_reg.bin', 'rb') as f_in:
    logging.info('Loading model and DV')
    (dv, model) = pickle.load(f_in)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    logging.info("Sucessfully prepared features")
    return features


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return float(preds[0])