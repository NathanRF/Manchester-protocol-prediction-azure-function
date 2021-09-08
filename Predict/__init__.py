import logging
import azure.functions as func

from tensorflow.keras import datasets, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.nn import relu, softmax
from matplotlib import pyplot as plt
import tensorflowjs as tfjs
import tensorflow as tf
from tensorflow.python.framework import convert_to_constants
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    logging.info(req.params)

    req_body = req.get_json()
    print(req_body)

    # Get data from request
    age = float(req_body.get('Age'))
    presenting_problem = float(req_body.get('PresentingProblem'))
    positive_discriminator = float(req_body.get('PositiveDiscriminator'))
    respiratory_rate = float(req_body.get('RespiratoryRate'))
    heart_rate = float(req_body.get('HeartRate'))
    oxigen_saturation = float(req_body.get('OxygenSaturation'))
    temperature = float(req_body.get('Temperature'))

    print(age, presenting_problem, positive_discriminator,
          respiratory_rate, heart_rate, oxigen_saturation, temperature)

    # Load the model
    model = load_model('Model\model.h5')

    # Predict
    parameters = np.array([[age, respiratory_rate, heart_rate, temperature,
                          oxigen_saturation, positive_discriminator, presenting_problem]])
    result = model.predict(parameters)

    print(result)

    # Get result name
    if np.argmax(result) == 0:
        result_name = 'NonUrgent'
    elif np.argmax(result) == 1:
        result_name = 'Standard'
    elif np.argmax(result) == 2:
        result_name = 'Urgent'
    elif np.argmax(result) == 3:
        result_name = 'VeryUrgent'
    elif np.argmax(result) == 4:
        result_name = 'Emergent'

    # Return result
    return func.HttpResponse(result_name)
