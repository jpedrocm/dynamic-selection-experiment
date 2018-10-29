###############################################################################

from sklearn.neighbors import KNeighborsClassifier

class TSTBClassifier():

	def __init__(self, pool_classifiers, with_IH=False, IH_rate=0.0, k=7, 
				 random_state=None, n_jobs=-1):

		self.IH_rate = IH_rate,
		self.with_IH = with_IH,
		self.large_k = k,
		self.small_k = self._create_small_k(),
		self.random_state=random_state,
		self.n_jobs = n_jobs,
		self.easy_estimator = self._create_easy_estimator(),
		self.hard_estimator_left = self._create_hard_estimator_left(),
		self.hard_estimator_right = pool_classifiers,
		self.dsel_samples = None,
		self.dsel_labels = None

	def _create_easy_estimator(self):
		return self._create_knn(self.large_k, 'uniform')

	def _create_small_k(self):
		new_k = self.large_k/2
		new_k += int(new_k%2==0)
		return new_k

	def _create_hard_estimator_left(self):
		return self._create_knn(self.small_k, 'distance')

	def _create_knn(self, k, weights):
		knn = KNeighborsClassifier(n_neighbors=k, weights=weights,
								   n_jobs=self.n_jobs)
		return knn

	def fit(self, X, y):
		self.dsel_samples = X
		self.dsel_labels = y

	def predict(self, X):
		pass