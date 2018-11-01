# Homework 3

![python](https://img.shields.io/badge/python-3.7-blue.svg)
![status](https://img.shields.io/badge/status-complete-brightgreen.svg)
![license](https://img.shields.io/badge/license-MIT-green.svg)

This is the third homework for the Multiple Classifiers System's class. The project was forked and adapted from the [first homework](https://github.com/jpedrocm/jpcm-lista1-codigo).

## Description

The goal of this homework is to perform an experiment comparing two different dynamic classifier selection strategies, two different dynamic ensemble selection strategies and a two-level classifier system which takes into account the hardness of examples. Bagging was chosen for generating pools of 100 Perceptrons, which are combined with hard voting. Metrics are collected for each fold in a 10-fold cross-validation setting. They include accuracy, f-measure, AUC and g-mean. Means and standard deviations of these metrics are calculated in order to analyze the results for two datasets.

## Getting Started

### Requirements

* [Python](https://www.python.org/) >= 3.7.0
* [NumPy](http://www.numpy.org/) >= 1.15.2
* [SciPy](https://www.scipy.org/) >= 1.1.0
* [pandas](https://pandas.pydata.org/) >= 0.23.4
* [scikit-learn](http://scikit-learn.org/stable/) >= 0.20.0
* [DESlib](https://github.com/Menelau/DESlib) >= 0.3.dev


### Installing

* Clone this repository into your machine
* Download and install all the requirements listed above in the given order
* Download the CM1 and JM1 software defect prediction datasets in .arff format from the [Promise repository](http://promise.site.uottawa.ca/SERepository/datasets-page.html) and do not change their names
* Place both .arff files inside the data/ folder

### Reproducing

* Enter into the code/ folder in your local repository
* Run the experiment to produce every ensemble's predictions
```
python generate_predictions.py
```
* Generate all metric results
```
python generate_metrics.py
```
* Then, compare the scenarios wanted
```
python compare_scenarios.py [-f FILENAME] [-s SEPARATE] [-c1 COLUMN1] [-c2 COLUMN2]
```

## Project Structure

    .            
    ├── code                                  # Code files
    |   ├── compare_scenarios.py              # Compare metric results 
    │   ├── generate_metrics.py               # Generate metric results
    │   ├── generate_predictions.py           # Generate models predictions
    │   ├── two_stage_tiebreak_classifier.py  # Two stage ensemble based on instance hardness and tiebreaking rule
    │   └── utils.py                          # Utils functions
    ├── comparisons                           # Result comparison files
    ├── data                                  # Datasets files
    ├── metrics                               # Metrics files
    ├── predictions                           # Models predictions files
    ├── LICENSE.md
    └── README.md

## Author

* [jpedrocm](https://github.com/jpedrocm)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.