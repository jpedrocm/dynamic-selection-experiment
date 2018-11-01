###############################################################################

from collections import Counter

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from deslib.dcs import OLA, LCA

class TSTBClassifier():

	def __init__(self, pool_classifiers, selection_method,
				 tiebreak_estimator=None, tiebreak_fit=False, with_IH=False, 
				 IH_rate=0.0, k=7, DFP=True, random_state=None, n_jobs=-1):

		self.pool_classifiers = pool_classifiers
		self.selection_method = selection_method
		self.tiebreak_estimator = self._create_tiebreak_estimator(tiebreak_estimator)
		self.tiebreak_fit = tiebreak_fit
		self.IH_rate = IH_rate
		self.with_IH = with_IH
		self.large_k = int(k)
		self.small_k = self._create_small_k()
		self.dfp = DFP
		self.random_state=random_state
		self.n_jobs = n_jobs
		self.easy_estimator = self._create_easy_estimator()
		self.hard_estimator_right = self._create_hard_estimator(self.large_k)
		self.hard_estimator_left = self._create_hard_estimator(self.small_k)
		self.dsel_labels = None

	def _create_easy_estimator(self):
		return self._create_knn(self.large_k, 'uniform')

	def _create_knn(self, k, weights):
		knn = KNeighborsClassifier(n_neighbors=k, weights=weights,
								   n_jobs=self.n_jobs)
		return knn

	def _create_small_k(self):
		new_k = self.large_k//2
		new_k += int(new_k%2==0)
		return new_k

	def _create_hard_estimator(self, k):
		clf = None

		if self.selection_method == 'ola':
			clf = OLA(self.pool_classifiers, DFP=self.dfp, k=k)
		elif self.selection_method == 'lca':
			clf = LCA(self.pool_classifiers, DFP=self.dfp, k=k)
		else:
			raise ValueError("The chosen selection method is not available")

		return clf

	def _create_tiebreak_estimator(self, tiebreak_estimator):

		if tiebreak_estimator == None:
			return self.pool_classifiers
		else:
			return tiebreak_estimator

	def fit(self, X, y):
		self.dsel_labels = y
		self.easy_estimator.fit(X, y)
		self.hard_estimator_left.fit(X, y)
		self.hard_estimator_right.fit(X, y)
		if self.tiebreak_fit == True:
			self.tiebreak_estimator.fit(X, y)

	def _get_kdn(self, k_labels, instance_label):
		return sum([l != instance_label for l in k_labels])

	def _get_IHs(self, X, k, instance_labels):
		idxs_matrix = self.easy_estimator.kneighbors(X=X, n_neighbors=k,
		                                             return_distance=False)

		neighbors_labels = [[self.dsel_labels[i] for i in idxs_matrix[j]] for j in range(len(idxs_matrix))]
		kdns = [self._get_kdn(neighbors_labels[i], instance_labels[i]) for i in range(len(neighbors_labels))]
		return np.array(kdns)/float(k)

	def _hard_stage_instance_predict(self, left_label, right_label, tb_label):
		if left_label != right_label:
			return tb_label
		else:
			return left_label

	def _select_stage(self, x, ih, easy_label, left_label, right_label, tb_label):
		return easy_label if ih <= self.IH_rate else self._hard_stage_instance_predict(
																			left_label, 
			                                                                right_label, 
			                                                                tb_label)

	def _two_stages_predict(self, X):
		easy_labels = self.easy_estimator.predict(X)
		IHs = self._get_IHs(X, self.large_k, easy_labels)
		left_labels = self.hard_estimator_left.predict(X)
		right_labels = self.hard_estimator_right.predict(X)
		tb_labels = self.tiebreak_estimator.predict(X)

		return [self._select_stage(X[i], IHs[i], easy_labels[i], left_labels[i], 
			                       right_labels[i], tb_labels[i]) for i in range(len(easy_labels))]

	def _hard_stage_predict(self, X):
		left_labels = self.hard_estimator_left.predict(X)
		right_labels = self.hard_estimator_right.predict(X)
		tb_labels = self.tiebreak_estimator.predict(X)

		return [self._hard_stage_instance_predict(left_labels[i], right_labels[i], 
			                          tb_labels[i]) for i in range(len(tb_labels))]

	def predict(self, X):
		predictions = []

		if self.with_IH == True:
			predictions = self._two_stages_predict(X)
		else:
			predictions = self._hard_stage_predict(X)

		return np.array(predictions)