##########
CODE FILES
##########

* utils.py
This file contains different utility functions used by scripts in the code base

* feature_eningeer.py
This is the first file you should run. Perform basic feature engineering (e.g add weekday from the date) and converts the raw dataframe into processed ones.
Output : 2 .csv called "emotions.csv" and "tweets.csv" respectively

* klasses.py
This file contains the classes Cascade and CascadeData
The class "Cascade" is used to store and manipulate the data w.r.t a single cascade
The class "CascadeData" inherits the default pytorch Dataset class. It stores all the ids of the cascades to learn from and keep track of sampling, data aumentation, and "variant" of the df (i.e. not cropped, 2000_tweets, ...)

* cascade_generator.py
This is the second file you should run. It converts the csv with the twets and emotions in Train and Test data, each of them including, for every crop in terms of time, numbers and no crop, one csv called "grouped[_CROP][_TEST].csv" and a certain number of cascades , as identified by their id e.g. "11111[_CROP][_TEST].pt" saved as Cascade data structure (from klasses). It also performs all the feature engineering that needs to be done AFTER the train/test split. (e.g. normalization ...) and calculate the cascade wide statistics
Output : 
- 24 csvs : grouped[_crop][_test] (12 crops including no crop per test and train respectively)
- 12 * 2156 cascades, of which 12 * 1832 train and 12 * 324 test 

* trees.py
Contains the trees and the necessary classes to build them. These are :
- ChildSumTreeLSTMCell: class for childsum lstm cell + related methods, inherits from nn.Module
- TreeLSTM: class for tree lstm ; encapsulates ChildSumTreeLSTMCell, inherits from nn.Module
- BiDiTreeLSTM: class for bi-directional tree lstm ; encapsulates TWO ChildSumTreeLSTMCell, inherits from nn.Module
- DeepTreeLSTM: class for Cascade-LSTM ; encapsulates BiDiTreeLSTM or TreeLSTM depending on settings + has an additional neural network (nn.Sequential from pytorch)

* sampler.py
Contains the samplers to apply as well as the class IterativeSampler
- IterativeSampler: allow to apply a sequence of samplers while keeping the indexes in the initial matrix of the data points that remains after the full process.

* main.py
Actually train the model. Split ids on train/test or train/validation depending on parameters. creates the DataLoaders, build the cascade lstm in agreement with the parameters. At every "new best model" stores model weights, as well as results. When finishes, store a single line with model parameters and performance in the logs for post-evaluation.
