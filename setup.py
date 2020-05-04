from distutils.core import setup
long_description = '''
Kydavra is a python sci-kit learn inspired package for feature selection. It used some statistical methods to extract from pure pandas Data Frames the columns that are related to column that your model should predict.\n
This version of kydavra has the next methods of feature selection:\n
1) ANOVA test selector (ANOVASelector).\n
2) Chi squared selector (ChiSquaredSelector).\n
3) Genetic Algorithm selector (GeneticAlgorithmSelector).\n
4) Kendall Correlation selector (KendallCorrelationSelector).\n
5) Lasso selector (LassoSelector).\n
6) Pearson Correlation selector (PearsonCorrelationSelector).\n
7) Point-Biserial selector (PointBiserialCorrSelector).\n
8) P-value selector (PValueSelector).\n
9) Spearman Correlation selector (SpermanCorrelationSelector).\n
All these methods takes the pandas Data Frame and y column to select from remained columns in the Data Frame.\n

How to use kydavra\n
To use selector from kydavra you should just import the selector from kydavra in the following framework:\n
```from kydavra import <class name>```\n
class names are written above in parantheses.\n
Next create a object of this algorithm (I will use p-value method as an example).\n
```method = PValueSelector()```\n
To get the best feature on the opinion of the method you should use the 'select' function, using as parameters the pandas Data Frame and the column that you want your model to predict.\n
```selected_columns = method.select(df, 'target')```\n
Returned value is a list of columns selected by the algorithm.\n

Some methods could plot the process of selecting the best features.\n
In these methods dotted are features that wasn't selected by the method.\n
*ChiSquaredSelector*\n
```method.plot_chi2()```\n
For ploting and\n
```method.plot_chi2(save=True, file_path='FILE/PATH.png')```\n
and\n
```method.plot_p_value()```\n
for ploting the p-values.\n
*LassoSelector*\n
```method.plot_process()```\n
also you can save the plot using the same parameters.\n
*PValueSelector*\n
```method.plot_process()```\n

Some advices.\n
1) Use ChiSquaredSelector for categorical features.\n
2) Use LassoSelector and PValueSelector for regression problems.\n
3) Use PointBiserialCorrSelector for binary classification problems.\n

With love from Sigmoid.\n

We are open for feedback. Please send your impression to vpapaluta06@gmail.com\n

'''
setup(
  name = 'kydavra',
  packages = ['kydavra'],
  version = '0.1.7',
  license='MIT',
  description = 'Kydavra is a sci-kit learn inspired python library with feature selection methods for Data Science and Macine Learning Model development',
  long_description=long_description,
  author = 'SigmoidAI - Păpăluță Vasile',
  author_email = 'vpapaluta06@gmail.com',
  url = 'https://github.com/user/reponame',
  download_url = 'https://github.com/ScienceKot/kydavra/archive/v1.0.tar.gz',    # I explain this later on
  keywords = ['ml', 'machine learning', 'feature selection', 'python'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'pandas',
          'scikit-learn',
          'statsmodels',
          'matplotlib',
          'seaborn'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Framework :: Jupyter',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)