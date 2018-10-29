###############################################################################

from collections import Counter

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from deslib.dcs import OLA, LCA

class TSTBClassifier():

	def __init__(self, pool_classifiers, selection_method, with_IH=False, IH_rate=0.0, 
		         k=7, random_state=None, n_jobs=-1):

		self.IH_rate = IH_rate
		self.with_IH = with_IH
		self.large_k = int(k)
		self.selection_method = selection_method
		self.small_k = self._create_small_k()
		self.random_state=random_state
		self.n_jobs = n_jobs
		self.easy_estimator = self._create_easy_estimator()
		self.hard_estimator_right = pool_classifiers
		self.hard_estimator_left = self._create_hard_estimator_left()
		self.tiebreaking_estimator = self._create_tiebreak_estimator(self.large_k)
		self.dsel_labels = None

	def _create_easy_estimator(self):
		return self._create_knn(self.large_k, 'uniform')

	def _create_small_k(self):
		new_k = self.large_k//2
		new_k += int(new_k%2==0)
		return new_k

	def _create_hard_estimator_left(self):
		#return self._create_knn(self.small_k, 'uniform')
		return self._create_tiebreak_estimator(self.small_k)

	def _create_tiebreak_estimator(self, k):
		clf = None

		if self.selection_method == 'ola':
			clf = OLA(self.hard_estimator_right, DFP = True, k = k)
		elif self.selection_method == 'lca':
			clf = LCA(self.hard_estimator_right, DFP = True, k = k)

		return clf

	def _create_knn(self, k, weights):
		knn = KNeighborsClassifier(n_neighbors=k, weights=weights,
								   n_jobs=self.n_jobs)
		return knn

	def fit(self, X, y):
		self.dsel_labels = y
		self.easy_estimator.fit(X, y)
		self.hard_estimator_left.fit(X, y)
		self.tiebreaking_estimator.fit(X, y)

	def _get_IH(self, x, k, instance_label):
		idxs = self.easy_estimator.kneighbors(X=[x], n_neighbors=k,
		                                      return_distance=False)
		neighbors_labels = [self.dsel_labels[i] for i in idxs[0]]
		kdn = sum([l!=instance_label for l in neighbors_labels])
		return kdn/float(k)

	def _predict_hard_stage(self, x):
		left_label = self.hard_estimator_left.predict([x])[0]
		right_label = self.hard_estimator_right.predict([x])[0]

		if left_label != right_label:
			return self.tiebreaking_estimator.predict([x])[0]
		else:
			return left_label

	def _two_stage_predict(self, x):
		easy_label = self.easy_estimator.predict([x])[0]
		easy_IH = self._get_IH(x, self.large_k, easy_label)

		if easy_IH <= self.IH_rate:
			return easy_label
		else:
			return self._predict_hard_stage(x)


	def predict(self, X):
		
		predictions = []

		if self.with_IH == True:
			predictions = [self._two_stage_predict(xi) for xi in X]
		else:
			predictions = [self._predict_hard_stage(xi) for xi in X]

		return np.array(predictions)