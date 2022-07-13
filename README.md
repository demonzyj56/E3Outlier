# Demo implementation of E3Outlier for Unsupervised Outlier Detection

## Introduction
This repository provides the implementation of discriminative E3Outlier, an effective and end-to-end method for the unsupervised outlier detection (UOD) task. UOD aims to directly detect outliers from a contaminated unlabeled dataset in a transductive manner, without using any labeled data (e.g. a labeled training set with pure normal data/inliers).

## Requirements 
- Python 3.6
- PyTorch 0.4.1 (GPU)
- Keras 2.2.0 
- Tensorflow 1.8.0 (GPU)
- sklearn 0.19.1
 

## Usage

To run E3Outlier with default settings, simply run the following command:

```bash
python outlier_experiments.py
```

This will automatically run UOD methods on all datasets (`MNIST`, `Fashion-MNIST`, `SVHN`, `CIFAR10` and `CIFAR100`). On each dataset the experiment will be conducted with 5 outlier ratios: 0.05, 0.1, 0.15, 0.2 and 0.25.

After learning, the prediction scores and ground truth labels are saved to an `npz` file. To obtain the UOD result for a specific algorithm, run ```evaluate_roc_auc.py``` for evaluation using *Area under the ROC curve (AUROC)*, or ```evaluate_pr_auc.py``` for evaluation using *Area under the PR curve (AUPR)*. Example usage:

```bash
# AUROC of E3Outlier on CIFAR10 with outlier ratio 0.1
python evaluate_roc_auc.py --dataset cifar10 --algo_name e3outlier-0.1

# AUPR of E3Outlier on MNIST with outlier ratio 0.25 and inliers as the postive class
python evaluate_pr_auc.py --dataset mnist --algo_name e3outlier-0.25 --postive inliers
```

## License

E3Outlier is released under the MIT License.
