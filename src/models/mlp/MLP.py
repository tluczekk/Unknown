from sklearn import neural_network
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
import numpy as np

class MLP:
    def __init__(self) -> None:
        self.mlp = neural_network.MLPClassifier(
            hidden_layer_sizes=(100,),
            learning_rate_init=0.001,
            max_iter=5000,
            early_stopping=True,
            validation_fraction=0.1,
            tol=1e-4,
            n_iter_no_change=10,
            random_state=0,
        )

    def get_estimator(self, training_samples, training_labels):
        # Getting the best estimator using 5-fold cross validation
        cv_results = cross_validate(
            self.mlp, 
            training_samples, 
            training_labels,
            scoring="accuracy",
            return_estimator=True
        )
        estimator_to_return = np.argmax(cv_results['test_score'])
        estimator = cv_results['estimator'][estimator_to_return]
        return estimator