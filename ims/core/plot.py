import pickle
import pandas as pd
# plt error 우회
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data(file_path):
    # import result for train/test loss
    df = pd.read_pickle(file_path)
    return df


def get_loss_graph(df, fig_size, label_size, font_size, save_path, png_name):
    # generate train/test graph
    rng = [ x+1 for x in range(df.shape[0])]
    plt.figure(figsize=fig_size)
    plt.plot(rng, df["tr_losses"], marker = '.', label = "train-loss")
    plt.plot(rng, df["te_losses"], 'r', label = "test-loss")
    plt.ylabel("loss", size = label_size)
    plt.xlabel("epoch", size = label_size)
    plt.legend(fontsize = font_size)
    plt.savefig(save_path + png_name)
    print("saved graph_loss..!")
    #plt.show()
    plt.close()


def get_pred_graph(actual, pred, fig_size, label_size, font_size, save_path, png_name):
    # generate actual/
    rng = [ x+1 for x in range(len(actual))]
    plt.figure(figsize=fig_size)
    plt.plot(rng, actual, marker = '.', label = "actual")
    plt.plot(rng, pred, 'r', label = "perdiction")
    plt.ylabel("issues", size = label_size)
    plt.xlabel("period", size = label_size)
    plt.legend(fontsize = font_size)
    plt.savefig(save_path + png_name)
    print("saved graph_pred..!")
    #plt.show()
    plt.close()
