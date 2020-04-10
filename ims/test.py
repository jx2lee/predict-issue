from core.model import *
from core.pred import *
from core.plot import *
from core.var import *

# test
input_, dev_set = get_issues(period_size = PERIOD_SIZE)
prediction = predict_issues(prediction_length = PERIOD_SIZE, new_input = input_, checkpoint_path = CHECKPOINT_PATH)
print('prediction : {}'.format(prediction))
print('actual : {}'.format(dev_set[0]))
print('prediction finished..!')

# create plot
result = load_data(DF_PATH)
get_loss_graph(df = result, fig_size = FIG_SIZE, label_size = LABEL_SIZE, font_size = FONT_SIZE,
               save_path = RESULT_PATH, png_name = "graph_loss.png")
get_pred_graph(actual = dev_set[0], pred = prediction, fig_size = FIG_SIZE, label_size = LABEL_SIZE, font_size = FONT_SIZE,
               save_path = RESULT_PATH, png_name = "graph_pred.png")
