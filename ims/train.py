from core.utils import *
from core.model import *
from core.var import *

# dataframe
df = import_data(csv_path = CSV_PATH + CSV_NAME, convert_type = ['Closed Date', 'Registered date'], index_col = 'Registered date')
# train / test set
train_X, train_y, test_X, test_y, num_train, num_test = split_data(df, n_train_time = 365*12, period_size = PERIOD_SIZE)
# batch generator
generator_train = batch_generator(train_X, train_y, num_train = num_train, batch_size = BATCH_SIZE, period_size = PERIOD_SIZE, output_size = OUTPUT_SIZE)
generator_test = batch_generator(test_X, test_y, num_train = num_test, batch_size = BATCH_SIZE, period_size = PERIOD_SIZE, output_size = OUTPUT_SIZE)
# model
g = build_model(period_size = PERIOD_SIZE, output_size = OUTPUT_SIZE, state_size = STATE_SIZE, batch_size = BATCH_SIZE,
                lstm_size = [STATE_SIZE, STATE_SIZE], dropout_prob = DROPOUT_PROB)
# train
_, _ = train_model(g, batch_size = BATCH_SIZE, data_size = [num_train, num_test], epoch = EPOCH, batch_generator = [generator_train, generator_test],
                   checkpoint_path = CHECKPOINT_PATH, df_path = DF_PATH)
print('training finished..!')