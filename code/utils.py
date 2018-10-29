###############################################################################
from functools import partial
from math import sqrt
from copy import deepcopy
import operator, sys

import json
import pandas as pd
import numpy as np
from scipy.io import arff

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score

from deslib.dcs import OLA, LCA
from deslib.des import KNORAE, KNORAU

from two_stage_tiebreak_classifier import TSTBClassifier


def load_experiment_configuration():
	BAGGING_PERCENTAGE = 0.5
	N_JOBS = -1
	K_COMPETENCE = 7

	config = {
	"num_folds": 10,
	"pool_size": 100,
	"k_competence": K_COMPETENCE,
	"base_classifier": partial(Perceptron, max_iter = 40, tol = 0.001,
		                       penalty = None, n_jobs = N_JOBS),
	"generation_strategy": partial(BaggingClassifier, 
		                           max_samples = BAGGING_PERCENTAGE,
		                           n_jobs = -1),
	"selection_strategies": _create_selection_strategies(K_COMPETENCE)
	}

	return config

def _create_selection_strategies(k_competence):
	return [("F-KNU", "DES", partial(KNORAU, DFP=True, k=k_competence)),
	        ("F-KNE", "DES", partial(KNORAE, DFP=True, k=k_competence)),
	        ("F-OLA", "DCS", partial(OLA, DFP=True, k=k_competence)),
	        ("F-LCA", "DCS", partial(LCA, DFP=True, k=k_competence)),
	        ("Two-Stage.0IH", "Hybrid", partial(TSTBClassifier, k=k_competence,
	                                        with_IH=True, IH_rate=0.0)),
	        ("Two-Stage.2IH", "Hybrid", partial(TSTBClassifier, k=k_competence,
	                                        with_IH=True, IH_rate=0.2)),
	        ("Two-Stage.4IH", "Hybrid", partial(TSTBClassifier, k=k_competence,
	                                        with_IH=True, IH_rate=0.4))] # [KNN] pra faceis : [Hard Voting(Weighted KNN + Ensemble) ELSE voto de minerva] pra dificeis

def scale_data(train_instances, validation_instances, test_instances):
	scaler = StandardScaler()
	train_instances = scaler.fit_transform(train_instances)
	validation_instances = scaler.transform(validation_instances)
	test_instances = scaler.transform(test_instances)
	return train_instances, validation_instances, test_instances

def load_datasets_filenames():
	filenames = ["cm1", "jm1"]
	return filenames

def load_dataset(set_filename):
	SET_PATH = "../data/"
	FILETYPE = ".arff"
	full_filepath = SET_PATH + set_filename + FILETYPE

	data, _ = arff.loadarff(full_filepath)

	dataframe = pd.DataFrame(data)
	dataframe.dropna(inplace=True)

	gold_labels = pd.DataFrame(dataframe["defects"])
	instances = dataframe.drop(columns = "defects")

	gold_labels = (gold_labels["defects"] == b'true').astype(int)

	return instances, gold_labels

def save_predictions(data):
	with open('../predictions/all_predictions.json', 'w') as outfile:
		json.dump(data, outfile)

def load_predictions_data():
	with open('../predictions/all_predictions.json', 'r') as outfile:
		return json.load(outfile)

def _error_score(gold_labels, predicted_labels):
	return 1 - accuracy_score(gold_labels, predicted_labels)

def _g1_score(gold_labels, predicted_labels, average):
	precision = precision_score(gold_labels, predicted_labels, average=average)
	recall = recall_score(gold_labels, predicted_labels, average=average)
	return sqrt(precision*recall)

def _calculate_metrics(gold_labels, predicted_labels):

	metrics = {}
	metrics["auc_roc"] = roc_auc_score(gold_labels, predicted_labels, average='macro')
	metrics["g1"] = _g1_score(gold_labels, predicted_labels, average='macro')
	metrics["f1"] = f1_score(gold_labels, predicted_labels, average='macro')
	metrics["acc"] = accuracy_score(gold_labels, predicted_labels)

	return metrics

def generate_metrics(predictions_dict):
	metrics = {}

	for set_name, set_dict in predictions_dict.items():
		metrics[set_name] = {}

		for fold, fold_dict in set_dict.items():

			gold_labels = fold_dict["gold_labels"]
			del fold_dict["gold_labels"]

			for strategy, data in fold_dict.items():

				fold_metrics = _calculate_metrics(gold_labels, data[0])

				if strategy not in metrics[set_name].keys():
				    metrics[set_name][strategy] = {"type": data[1], "metrics": [fold_metrics]}
				else:
					metrics[set_name][strategy]["metrics"].append(fold_metrics)

	return metrics

def _summarize_metrics_folds(metrics_folds):
	summary = {}
	metric_names = metrics_folds[0].keys()

	for metric_name in metric_names:
		scores = [metrics_folds[i][metric_name] for i in range(len(metrics_folds))]
		summary[metric_name] = [np.mean(scores), np.std(scores)]

	return summary

def summarize_metrics_folds(metrics_dict):

	summary = deepcopy(metrics_dict)

	for set_name, set_dict in metrics_dict.items():
		for strategy_name, data_folds in set_dict.items():
			cur_metrics_summary = _summarize_metrics_folds(data_folds["metrics"])
			summary[set_name][strategy_name] = {"metrics": cur_metrics_summary,
			                                   "type": data_folds["type"]}

	return summary

def pandanize_summary(summary):

	df = pd.DataFrame(columns = ['set', 'strategy', 'type',
	                  'mean_auc_roc', 'std_auc_roc', 'mean_acc', 'std_acc',
	                  'mean_f1', 'std_f1', 'mean_g1', 'std_g1'])

	for set_name, set_dict in summary.items():
		for strategy, summary_folds in set_dict.items():
			df_folds = pd.DataFrame(_unfilled_row(3, 8),
				                    columns = df.columns)
			_fill_dataframe_folds(df_folds, summary_folds, set_name,
				                  strategy)
			df = df.append(df_folds)

	return df.reset_index(drop = True)

def _unfilled_row(nb_str_columns, nb_float_columns):
	row = [" " for i in range(nb_str_columns)]
	row.extend([0.0 for j in range(nb_float_columns)])
	return [row]

def _fill_dataframe_folds(df, summary, set_name, strategy):
	df.at[0, "set"] = set_name
	df.at[0, "strategy"] = strategy
	df.at[0, "type"] = summary["type"]
	return _fill_dataframe_metrics(df, summary["metrics"])

def _fill_dataframe_metrics(df, summary):
	for key, metrics in summary.items():
		df.at[0, "mean_" + key] = metrics[0]
		df.at[0, "std_" + key] = metrics[1]
	return df

def save_pandas_summary(df):
	pd.to_pickle(df, '../metrics/metrics_summary.pkl')

def read_pandas_summary():
	return pd.read_pickle('../metrics/metrics_summary.pkl')

def separate_pandas_summary(df, separate_sets):
	dfs = []

	if separate_sets == True:
		sets = df["set"].unique()
		for set_name in sets:
			dfs.append(df.loc[df["set"]==set_name])
	else:
		dfs.append(df)

	return dfs

def write_comparison(dfs, focus_columns, filename):

	with open('../comparisons/'+ filename + '.txt', "w") as outfile:
		for df_set in dfs:
			if len(dfs) == 1:
				outfile.write("\n\nDATASET: Mixed\n")
			else:
				outfile.write("\n\nDATASET: " + df_set.iat[0,0] + "\n")
			outfile.write("Mean of metrics\n")
			outfile.write(df_set.groupby(by=focus_columns).mean().to_string())
			outfile.write("\n\nStd of metrics\n")
			outfile.write(df_set.groupby(by=focus_columns).std().fillna(0).to_string())
			outfile.write("\n")
			outfile.write("-------------------------------------------------")

def bool_str(s):

    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')

    return s == 'True'