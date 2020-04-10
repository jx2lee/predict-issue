# **predict IMS issues**   

* Model to predict number of IMS registrations per day<br>
* Used to demonstrate HyperData 8 scenarios

## **Model description**<br>
**LSTM**<br>
> num of Layers : 1<br>
> Unit-Size : 256<br>
> Dropout-Prob : 0.3<br>
>> Epoch : 1000<br>
>> Batch-Size : 30<br>
>> Priod-Size : 20<br>

## **Code descroption**

> train.py : train model<br>
> test.py : predict & generate graph<br>
>> core/var.py : set parameters (*modify parameters here*) <br>
>> core/utils.py : import data & preprocess data<br>
>> core/model.py : set model using tensorflow<br>
>> core/pred.py : code for prediction using model<br>
>> core/plot.py : code for generating plot using matplotlib<br>

## **Run**

    $ python train.py
    $ python test.py
    

## **Result**
<img width="500" src="https://github.com/jaejuning/pilot/blob/master/ims/res/graph_pred.png" alt="Prunus">

----------
2019.07.20 made by *jaejun.lee*
