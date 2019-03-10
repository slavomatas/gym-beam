import numpy as np
import pandas as pd

import multiprocessing
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.timeseries as ts

from os import path
from sklearn import preprocessing

from tensorflow.contrib.timeseries.python.timeseries import TimeSeriesRegressor, state_management
from tensorflow.python.estimator.run_config import RunConfig

from pyanalytics.pipeline.factors.tensorflow.lstm_estimator import LSTMModel

# import matplotlib
# matplotlib.use("TkAgg")  # Need Tk for interactive plots.

TIME_INDEX_FEATURE_NAME = 'time_index'
VALUE_FEATURE_NAME = 'values'
MULTI_THREADING = True

MODULE_PATH = path.dirname(__file__)

symbol = 'XLK'

# TRAIN_DATA_FILE = path.join(MODULE_PATH, "data/etf-train-data.csv")
# TEST_DATA_FILE = path.join(MODULE_PATH, "data/etf-test-data.csv")

# TRAIN_DATA_FILE = path.join(MODULE_PATH, "data/xlk-train-data.csv")
# TEST_DATA_FILE = path.join(MODULE_PATH, "data/xlk-test-data.csv")

# TRAIN_DATA_FILE = path.join(MODULE_PATH, "data/xlf-test-data.csv")
# TEST_DATA_FILE = path.join(MODULE_PATH, "data/xlf-test-data.csv")

# TRAIN_DATA_FILE = path.join(MODULE_PATH, "data/xlu-train-data.csv")
# TEST_DATA_FILE = path.join(MODULE_PATH, "data/xlu-test-data.csv")

TRAIN_DATA_FILE = path.join(MODULE_PATH, 'data/' + symbol + '-train-data.csv')
TEST_DATA_FILE = path.join(MODULE_PATH, 'data/' + symbol + '-test-data.csv')

imputer = preprocessing.Imputer()
scaler = preprocessing.MinMaxScaler()


def compute_rmse(a, b):
    rmse = np.sqrt(np.sum(np.square(a - b)) / len(a))
    return round(rmse, 5)


def compute_mae(a, b):
    # mae = np.sqrt(np.sum(np.abs(a - b)) / len(a))
    mae = np.sum(np.abs(a - b)) / len(a)
    return round(mae, 5)


def create_input_fn(file_name, mode, header_lines=0, batch_size=None, windows_size=None, tail_count=None):
    input_data_df = pd.read_csv(file_name, skiprows=header_lines)

    # input_data_df[VALUE_FEATURE_NAME] = input_data_df[VALUE_FEATURE_NAME].shift(5)

    input_data_df = input_data_df.dropna()
    columns = input_data_df.columns

    features_df = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(input_data_df[columns[2:]])))
    features_df.columns = columns[2:]

    print("Dataset Size: {}".format(len(features_df)))

    data = {
        ts.TrainEvalFeatures.TIMES: input_data_df[TIME_INDEX_FEATURE_NAME],
        ts.TrainEvalFeatures.VALUES: input_data_df[VALUE_FEATURE_NAME],
    }

    data.update({
        feature: features_df[feature] for feature in features_df.columns
    })

    reader = ts.NumpyReader(data)

    num_threads = multiprocessing.cpu_count() if MULTI_THREADING else 1

    if mode == tf.estimator.ModeKeys.TRAIN:
        input_fn = tf.contrib.timeseries.RandomWindowInputFn(
            reader,
            batch_size=batch_size,
            window_size=windows_size,
            num_threads=num_threads
        )

    elif mode == tf.estimator.ModeKeys.EVAL:
        input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)

    return input_fn, features_df.columns


# Train and predict using a LSTM time series model
def tf_lstm_estimator_test(csv_file_name=TRAIN_DATA_FILE, training_steps=5000, estimator_config=None,
                           export_directory=None):
    tf.logging.set_verbosity(tf.logging.INFO)

    print('Device {}'.format(tf.test.gpu_device_name()))

    CHECKPOINT_STEPS = 200

    # Set hyper-params values
    hparams = tf.contrib.training.HParams(
        training_steps=training_steps,
        periodicities=[1],
        input_window_size=20,
        output_window_size=5,
        batch_size=4,
        loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS,  # NORMAL_LIKELIHOOD_LOSS | SQUARED_LOSS
        hidden_units=128,
        learning_rate=0.1
    )

    model_dir = 'trained_models/{}'.format(symbol + '-lstm-model-adagrad')

    run_config = RunConfig(
        save_checkpoints_steps=CHECKPOINT_STEPS,
        tf_random_seed=19831060,
        model_dir=model_dir
    )

    train_input_fn, feature_columns = create_input_fn(
        file_name=csv_file_name,
        mode=tf.estimator.ModeKeys.TRAIN,
        batch_size=hparams.batch_size,
        windows_size=hparams.input_window_size + hparams.output_window_size
    )

    # Exogenous features are not part of the loss, but can inform predictions.
    exogenous_feature_columns = [tf.feature_column.numeric_column(ex_feature_column) for ex_feature_column in
                                 feature_columns]

    optimizer = tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
    )

    # optimizer = tf.train.AdamOptimizer(0.001)

    # Create an LSTM estimator
    estimator = TimeSeriesRegressor(
        model=LSTMModel(num_features=1, num_units=hparams.hidden_units,  # num_features=1 for univariate time series
                        exogenous_feature_columns=exogenous_feature_columns),
        optimizer=optimizer, config=estimator_config,
        # Set state to be saved across windows.
        state_manager=state_management.ChainingStateManager(),
        model_dir=model_dir)

    estimator.train(input_fn=train_input_fn, steps=hparams.training_steps)

    """
    Evaluate the estimator after training
    """

    eval_input_fn, feature_columns = create_input_fn(
        file_name=TRAIN_DATA_FILE,
        mode=tf.estimator.ModeKeys.EVAL,
    )

    evaluation = estimator.evaluate(input_fn=eval_input_fn, steps=5)

    print("")
    print(evaluation.keys())
    print("")
    print("Evaluation Loss ({}) : {}".format(hparams.loss, evaluation['loss']))

    x_current = evaluation['times'][0]
    y_current_actual = evaluation['observed'][0].reshape(-1)
    y_current_estimated = evaluation['mean'][0].reshape(-1)

    rmse = compute_rmse(y_current_actual, y_current_estimated)
    mae = compute_mae(y_current_actual, y_current_estimated)
    print("Evaluation RMSE {}".format(rmse))
    print("Evaluation MAE {}".format(mae))

    predict(estimator, evaluation)

    '''
    plt.figure(figsize=(20, 10))

    plt.title(symbol + " - Returns Evaluation")
    plt.plot(x_current, y_current_actual, label='actual')
    plt.plot(x_current, y_current_estimated, label='estimated')
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.legend(loc=2)
    plt.show()
    '''


def predict(estimator, evaluation):
    """
    Test model predictions
    """

    df_test = pd.read_csv(TEST_DATA_FILE, names=['time_index', 'values'], header=0)
    # df_test = pd.read_csv(TEST_DATA_FILE)
    # print("Test Dataset Size: {}".format(len(df_test)))
    # print("")

    # Predict starting after the evaluation
    exogenous_features_df = pd.read_csv(TEST_DATA_FILE, skiprows=0)
    exogenous_features_df = exogenous_features_df.dropna()
    columns = exogenous_features_df.columns

    exogenous_features_df = pd.DataFrame(scaler.transform(imputer.transform(exogenous_features_df[columns[2:]])))
    exogenous_features_df.columns = columns[2:]

    print("Exogenous Features: {}".format(len(exogenous_features_df)))

    steps = 5

    for i, row in enumerate(exogenous_features_df[-50:].values):
        time_index = exogenous_features_df.index[i]

        predict_exogenous_features = {}
        predict_exogenous_features.update({
            feature: exogenous_features_df[feature][time_index:time_index + steps].reshape(1, -1) for feature in
            exogenous_features_df.columns
        })

        forecasts = estimator.predict(
            input_fn=ts.predict_continuation_input_fn(evaluation, steps=steps,
                                                      exogenous_features=predict_exogenous_features))

        forecasts = tuple(forecasts)[0]

        x_next = forecasts['times']

        y_next_forecast = forecasts['mean']
        y_next_actual = df_test['values'][time_index:time_index + steps].reshape(-1, 1)

        print("y_next_forecast {} y_next_actual {}".format(y_next_forecast, y_next_actual))

        rmse = compute_rmse(y_next_actual, y_next_forecast)
        mae = compute_mae(y_next_actual, y_next_forecast)

        print("Forecast Steps {}: RMSE {} - MAE {}".format(steps, rmse, mae))


if __name__ == "__main__":
    # ray.init()
    # ray.get(tf_lstm_estimator_test.remote())
    tf_lstm_estimator_test()
