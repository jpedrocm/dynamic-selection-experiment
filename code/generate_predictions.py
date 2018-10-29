###############################################################################
import numpy as np
import random as rn

#DO NOT CHANGE THIS
np.random.seed(2577)
rn.seed(3581)
###################

from utils import load_datasets_filenames, load_experiment_configuration
from utils import load_dataset, save_predictions
from utils import scale_data

from sklearn.model_selection import StratifiedKFold, train_test_split



if __name__ == "__main__":

	print("Step 1 - Loading configurations")

	datasets_filenames = load_datasets_filenames()
	config = load_experiment_configuration()
	predictions = {}
	exp = 1

	print("Step 2 - Starting experiment")

	for dataset_name in datasets_filenames:
		instances, gold_labels = load_dataset(dataset_name)
		skfold = StratifiedKFold(n_splits = config["num_folds"],
			                     shuffle = True)

		predictions[dataset_name] = {}

		for fold, division in enumerate(skfold.split(X=instances, y=gold_labels), 1):
			test_idxs = division[1]
			test_instances = instances.iloc[test_idxs].values
			test_gold_labels = gold_labels.iloc[test_idxs].values.ravel()

			predictions[dataset_name][fold] = {}
			predictions[dataset_name][fold]["gold_labels"] = test_gold_labels.tolist()

			dev_idxs = division[0]
			dev_instances = instances.iloc[dev_idxs]
			dev_gold_labels = gold_labels.iloc[dev_idxs]

			dev_tuple = train_test_split(dev_idxs, test_size = 0.2223, shuffle=True, 
				                         stratify=dev_gold_labels)

			train_idxs = dev_tuple[0]
			train_instances = instances.iloc[train_idxs].values
			train_gold_labels = gold_labels.iloc[train_idxs].values.ravel()
			val_idxs = dev_tuple[1]
			validation_instances = instances.iloc[val_idxs].values
			validation_gold_labels = gold_labels.iloc[val_idxs].values.ravel()

			scaled_instances = scale_data(train_instances, validation_instances, test_instances)
			train_instances = scaled_instances[0]
			validation_instances = scaled_instances[1]
			test_instances = scaled_instances[2]

			base_clf = config["base_classifier"]()
			clf_pool = config["generation_strategy"](base_clf, config["pool_size"])
			clf_pool.fit(train_instances, train_gold_labels)

			for strategy_name, strategy_type, selection_func in config["selection_strategies"]:

				ds_clf = selection_func(pool_classifiers=clf_pool)
				ds_clf.fit(validation_instances, validation_gold_labels)

				cur_predictions = ds_clf.predict(test_instances).astype(int)
				predictions[dataset_name][fold][strategy_name] = [cur_predictions.tolist(), strategy_type]

				print("Experiment " + str(exp))
				exp+=1

	print("Step 2 - Finished experiment")

	print("Step 3 - Storing predictions")
	save_predictions(predictions)