import tensorflow as tf
import numpy as np
import pickle
import pandas as pd

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def build_model(
    period_size,
    output_size,
    state_size,
    batch_size,
    lstm_size,
    dropout_prob):
    
    reset_graph()
    # Placeholders
    X = tf.placeholder(tf.float32, shape = [None, period_size])
    y = tf.placeholder(tf.float32, shape = [None, output_size])
    keep_prob = tf.constant(1.0)
    
    # Reshape X
    _X = tf.reshape(X, shape = [-1, 1, period_size])
    
    # LSTM layers
    lstms = [tf.contrib.rnn.BasicLSTMCell(size) for size in lstm_size]
    drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob) for lstm in lstms]
    cell = tf.contrib.rnn.MultiRNNCell(drops, state_is_tuple=True)
    ##### test 부분에서 batch 영향을 받지 않기위해 init_state 삭제 #####
    lstm_outputs, _ = tf.nn.dynamic_rnn(cell, _X, dtype=tf.float32)
    
    # Output layer
    with tf.variable_scope('output'):
        W = tf.get_variable('W', [state_size, output_size])
        b = tf.get_variable('b', [output_size], initializer=tf.constant_initializer(0.0))
        
    preds = tf.matmul(tf.reshape(lstm_outputs, [-1, state_size]), W) + b
    loss = tf.losses.mean_squared_error(labels=y, predictions=preds)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    saver = tf.train.Saver()
    
    return { 'X': X,
             'y': y,
             'dropout': keep_prob,
             'preds': preds,
             'loss': loss,
             'train_step': train_step,
             'saver' : saver
    }

def train_model(graph, batch_size, data_size, epoch, batch_generator,
                checkpoint_path, df_path):
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        
        with tf.name_scope('summary'):
            tf.summary.scalar('Loss', graph['loss'])
            tf.summary.histogram('histogram_loss', graph['loss'])

        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(checkpoint_path+'/summaryGraph',
                                        sess.graph)
        
        try:
            os.mkdir(checkpoint_path)
        except:
            print('directory existed..!')
        
        step, losses = 0, 0
        tr_losses, te_losses = [], []
        current_epoch = 0
        
        while current_epoch < epoch:
            step += 1
            batch_X, batch_y = next(batch_generator[0])
            feed = {graph['X']: batch_X, graph['y']: batch_y}
            loss_, _ = sess.run([graph['loss'], graph['train_step']], feed_dict=feed)
            losses += loss_
            
            if step % 10 == 0:
                print('current_epoch : {}\t step : {}'.format(current_epoch, step))
            
            if step == int(data_size[0] / batch_size):
                current_epoch += 1
                tr_losses.append(losses / step)                
                step, losses = 0, 0
                
                #model save
                saver.save(sess, checkpoint_path+'/model.ckpt', global_step=current_epoch)

                #eval test set
                while step < int(data_size[1] / batch_size):
                    step += 1
                    batch_X, batch_y = next(batch_generator[1])
                    feed = {graph['X']: batch_X, graph['y']: batch_y}
                    loss_, _ = sess.run([graph['loss'], graph['train_step']],feed_dict=feed)
                    losses += loss_
                    
                te_losses.append(losses / step)
                step, losses = 0, 0
                print("loss after epoch", current_epoch, " - tr_losses : ", tr_losses[-1], "- te_losses : ", te_losses[-1])
    
    # result.df
    result_dic = {'tr_losses' : tr_losses, 'te_losses' : te_losses}
    result_df = pd.DataFrame(result_dic, columns=['tr_losses', 'te_losses'])
    
    with open(df_path, 'wb') as p:
        pickle.dump(result_df, p)

    return tr_losses, te_losses