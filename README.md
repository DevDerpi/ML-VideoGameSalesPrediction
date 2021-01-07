# ML-VideoGameSalesPrediction
Global video game sales prediction from year 2008 to 2014 approximately using linear regression and decision tree regression with manipulating min_sample_split hyperparameter to achieve higher  accuracy /lower overfitting
A video game is an electronic game that involves interaction with a user interface or input device 
Since the 2010s, the commercial importance of the video game industry has been increasing. The emerging markets are driving the growth of the industry. As of 2018, video games generated sales of US$134.9 billion annually worldwide , and were the third-largest segment in the U.S. entertainment market, behind broadcast and cable TV. 
A regression model is built , using a linear regression model and decision tree regression model to predict the global sales , accuracy is obtained, and observations are made comparing the accuracies of both models , and the results of tuning the hyperparameters
At first , the important python libraries are imported to use their classes in our machine : 
Numpy : numpy contains mathematical tools and is what we need to include any mathematical things in our code
Pandas : it is the best library used to import and manage datasets 
2. Then we have to read our dataset 
3. Label encoding : categorical variables are variables that contain categories  [Yes or No , country names , any thing else than numbers]
this is done because as we know ML is based on mathematical equations , so strings and categorical variables would cause a problem , so we need to convert it to a representation that is suitable for computation . 
4. Then the Feature vector and dependent variable vector are extracted
5.Splitting the data into Test and train sets with ratio 80% to 20%: 
6.Scaling the data : because the variables are not on the same scale , which may cause some issues in the ML model, because most ML Models are built on Euclidean distance
 
then > Building the linear regression model
Linear regression model observation : Accuracy in the training data with Linear Regression Model:  96.50328265371722 % , 
                                    : Accuracy in the test data with Linear Regression model 96.57734070600729 %
then > Building the decision tree regression model
Decision tree model hyper-parameters :  Hyperparameter tuning is searching the hyperparameter space for a set of values that will optimize the model architecture
. Hyperparameter tuning is also tricky in the sense that there is no direct way to calculate how a change in the hyperparameter value will reduce the loss of your model
min_samples_split: int, float, optional (default=2):
 The minimum number of samples required to split an internal node:
According to our search , hyperparameter tuning of decision trees the ideal min_samples_split values tend to be between 1 to 40 to control over-fitting.
Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for the tree. Too high values can also lead to under-fitting hence depending on the level of underfitting or overfitting, the values min_samples_split are tuned.

:: Decision Tree Model hyperparameters 
There are other Hyperparameters we can tune to change in the accuracy of our model such as : 
max_depth: int or None, optional (default=None) :The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
min_samples_leaf: int, float, optional (default=1) :The minimum number of samples required to be at a leaf node.

Decision tree regression model observation Before tuning : 
Accuracy in the training data with Decision Tree Regression model before tuning  :  99.82093756158373 %
Accuracy in the test data with Decision Tree Regression model before tuining :  90.00469762316926 %
Decision tree regression model observation after tuning  :
Accuracy in the training data with Decision Tree Regression model After tuning with min_samples_split=35 :  96.65344771887644 %
Accuracy in the test data with Decision Tree Regression model After tuning with min_samples_split=35 :  93.08361849779752 %


