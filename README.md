# kydavra
Kydavra is a python sci-kit learn inspired package for feature selection. It used some statistical methods to extract from pure pandas Data Frames the columns that are related to column that your model should predict.
This version of kydavra has the next methods of feature selection:
* ANOVA test selector (ANOVASelector).
* Chi squared selector (ChiSquaredSelector).
* Genetic Algorithm selector (GeneticAlgorithmSelector).
* Kendall Correlation selector (KendallCorrelationSelector).
* Lasso selector (LassoSelector).
* Pearson Correlation selector (PearsonCorrelationSelector).
* Point-Biserial selector (PointBiserialCorrSelector).
* P-value selector (PValueSelector).
* Spearman Correlation selector (SpermanCorrelationSelector).
* Shannon selector (ShannonSelector).
All these methods takes the pandas Data Frame and y column to select from remained columns in the Data Frame.

How to use kydavra\
To use selector from kydavra you should just import the selector from kydavra in the following framework:\
```from kydavra import <class name>```\
class names are written above in parantheses.\
Next create a object of this algorithm (I will use p-value method as an example).\
```method = PValueSelector()```\
To get the best feature on the opinion of the method you should use the 'select' function, using as parameters the pandas Data Frame and the column that you want your model to predict.\
```selected_columns = method.select(df, 'target')```\
Returned value is a list of columns selected by the algorithm.

Some methods could plot the process of selecting the best features.\
In these methods dotted are features that wasn't selected by the method.\
*ChiSquaredSelector*\
```method.plot_chi2()```\
For ploting and\
```method.plot_chi2(save=True, file_path='FILE/PATH.png')```\
and\
```method.plot_p_value()```\
for ploting the p-values.\
*LassoSelector*\
```method.plot_process()```\
also you can save the plot using the same parameters.\
*PValueSelector*\
```method.plot_process()```

Some advices.
* Use ChiSquaredSelector for categorical features.
* Use LassoSelector and PValueSelector for regression problems.
* Use PointBiserialCorrSelector for binary classification problems.

With love from Sigmoid.

We are open for feedback. Please send your impression to vpapaluta06@gmail.com
