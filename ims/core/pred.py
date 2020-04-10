import random
import tensorflow as tf
import numpy as np
import pickle
from core.utils import import_data
from core.model import build_model
from core.var import *


def get_issues(period_size):
    # 가장 최신의 period 만큼의 데이터를 뽑아온다.
    df = import_data(csv_path = CSV_PATH + CSV_NAME, convert_type = ['Closed Date', 'Registered date'], index_col = 'Registered date')
    issues = df.tail(period_size*2)[:period_size]
    issues = issues.values.reshape(1, period_size)
    # dev-set
    dev = df.tail(period_size).values.reshape(1, period_size)
    return issues, dev


def predict_issues(prediction_length, new_input, checkpoint_path):
    # recall model
    model = build_model(period_size = PERIOD_SIZE, output_size = OUTPUT_SIZE, state_size = STATE_SIZE, batch_size = BATCH_SIZE,
                        lstm_size = [STATE_SIZE, STATE_SIZE], dropout_prob = DROPOUT_PROB)
    
    saver = model['saver']
    prediction = model['preds']
    # recall checkpoints and get predictions
    with tf.Session() as sess:
        saved_path = tf.train.latest_checkpoint(checkpoint_path)
        saver.restore(sess, saved_path)
        predictions = []
        for i in range(prediction_length):
            _prediction = sess.run(prediction, feed_dict = {model['X'] : new_input[:, i:]})
            # append prediction to predictions(array -> list)
            predictions.append(int(_prediction.tolist()[0][0]))
            # append array
            new_input = np.append(new_input, _prediction.astype(np.int16)).reshape(1, -1)
    # reshape predictions
    fn_predictions = np.array(predictions)
    return fn_predictions

