# SVM Model

All experiments were conducted on the reduced training set. 
Of course it would make sense to use all available data, but training is extremly time consuming. This is why we chose to go for the smaller data sets in order to be able to run more experiments.

As a consequence we used crossvalidation during training to make the best use of our reduced training set.
The evaluation is done with accuracy, f1 and recall scores to get a bit more insights in the classification and where the errors happen.

## Linear Kernel experiments

First we do some experiments with SVM with a linear Kernel.
The gamma parameter does not have any effect on this kernel so we only experiment with the C parameter.
First we have tried to decrease the parameter C 0.1 between the experiments. This has not had any effect on the results.
All measurements always had an accuracy of 91.22%. So then we divided the C parameter by 10 for every result to more rapitly approach 0.
This has let to better results the smaler the C value got as this experiment sequence shows:

```
started training SVM model...\
done training SVM model!\
--------SVM(kernel=linear, C=1.0, gamma=scale)-------\
 acc=0.9122\
 f1=0.9122000000000001\
 recall=0.9122\
----------------------------\

started training SVM model...\
done training SVM model!\
--------SVM(kernel=linear, C=0.1, gamma=scale)-------\
 acc=0.9122\
 f1=0.9122000000000001\
 recall=0.9122\
----------------------------\

started training SVM model...\
done training SVM model!\
--------SVM(kernel=linear, C=0.01, gamma=scale)-------\
 acc=0.9122\
 f1=0.9122000000000001\
 recall=0.9122\
----------------------------\

started training SVM model...\
done training SVM model!\
--------SVM(kernel=linear, C=0.001, gamma=scale)-------\
 acc=0.9136\
 f1=0.9136\
 recall=0.9136\
----------------------------\

started training SVM model...\
done training SVM model!\
--------SVM(kernel=linear, C=0.0001, gamma=scale)-------\
 acc=0.9168666666666667\
 f1=0.9168666666666667\
 recall=0.9168666666666667\
----------------------------\

started training SVM model...\
done training SVM model!\
--------SVM(kernel=linear, C=1e-05, gamma=scale)-------\
 acc=0.9281333333333334\
 f1=0.9281333333333334\
 recall=0.9281333333333334\
----------------------------\
```

We got up to 92.81% accuracy scores. Since training is quite time consuming, we switched to a second kernel to see if we can get even better results:


## RBF Kernel experiments

The training setup stayed the same for this experiment series, but here we have another parameter to experiment with: gamma.
This gives us way more experiments to conduct: one for each combination of gamma and C value.

Since a C value of 1e-05 gave the best results, we started with this value and tried different multiples of 10 for gamma:

```
started training SVM model...\
done training SVM model!\
--------SVM(kernel=rbf, C=1e-05, gamma=100.0)-------\
 acc=0.11193333333333333\
 f1=0.11193333333333333\
 recall=0.11193333333333333\
----------------------------\

started training SVM model...\
done training SVM model!\
--------SVM(kernel=rbf, C=1e-05, gamma=10.0)-------\
 acc=0.11193333333333333\
 f1=0.11193333333333333\
 recall=0.11193333333333333\
----------------------------\

started training SVM model...\
done training SVM model!\
--------SVM(kernel=rbf, C=1e-05, gamma=1.0)-------\
 acc=0.11193333333333333\
 f1=0.11193333333333333\
 recall=0.11193333333333333\
----------------------------\

started training SVM model...\
done training SVM model!\
--------SVM(kernel=rbf, C=1e-05, gamma=0.1)-------\
 acc=0.11193333333333333\
 f1=0.11193333333333333\
 recall=0.11193333333333333\
----------------------------\

started training SVM model...\
done training SVM model!\
--------SVM(kernel=rbf, C=1e-05, gamma=0.01)-------\
 acc=0.11193333333333333\
 f1=0.11193333333333333\
 recall=0.11193333333333333\
----------------------------\
```

The results were very dissapointing and did not change for any gamma value.
So we chose to start over with a C value of 1 and tried the same values for gamma again before then dividing the C value by 10 and starting over.
Basically we gradually searched for the ideal C, gamma value combination.
The only problem was that expermients were even more time consuming than with the linear kernel.
The results were again dissapointing. After 7h of experiments we had always the same dissapointing accuracy scores of 11%:

```
started training SVM model...\
done training SVM model!\
--------SVM(kernel=rbf, C=1.0, gamma=100.0)-------\
 acc=0.11193333333333333\
 f1=0.11193333333333333\
 recall=0.11193333333333333\
----------------------------\

started training SVM model...\
done training SVM model!\
--------SVM(kernel=rbf, C=1.0, gamma=10.0)-------\
 acc=0.11193333333333333\
 f1=0.11193333333333333\
 recall=0.11193333333333333\
----------------------------\

started training SVM model...\
done training SVM model!\
--------SVM(kernel=rbf, C=1.0, gamma=1.0)-------\
 acc=0.11193333333333333\
 f1=0.11193333333333333\
 recall=0.11193333333333333\
----------------------------\

started training SVM model...\
done training SVM model!\
--------SVM(kernel=rbf, C=1.0, gamma=0.1)-------\
 acc=0.11193333333333333\
 f1=0.11193333333333333\
 recall=0.11193333333333333\
----------------------------\

started training SVM model...\
done training SVM model!\
--------SVM(kernel=rbf, C=0.1, gamma=100.0)-------\
 acc=0.11193333333333333\
 f1=0.11193333333333333\
 recall=0.11193333333333333\
----------------------------\

started training SVM model...\
done training SVM model!\
--------SVM(kernel=rbf, C=0.1, gamma=10.0)-------\
 acc=0.11193333333333333\
 f1=0.11193333333333333\
 recall=0.11193333333333333\
----------------------------\

```

further experiments are currently still running...
