# Effective End-to-end Unsupervised Outlier Detection via Inlier Priority of Discriminative Network
By Siqi Wang, Yijie Zeng, Xinwang Liu, En Zhu, Jianping Yin, Chuanfu Xu, Marius Kloft.  To appear in NeurIPS 2019.

## Introduction
This is the official implementation of the E3Outlier framework presented by "Effective End-to-end Unsupervised Outlier Detection via Inlier Priority of Discriminative Network".
The codes are used to reproduce experimental results of  E3Outlier and other unsupervised outlier detection (UOD) methods reported in the paper.

## Requirements
- Python 3.6
- PyTorch 0.4.1 (GPU)
- Keras 2.2.0 
- Tensorflow 1.8.0 (GPU)
- sklearn 0.19.1
 

## Usage

To obtain the results of E3Outlier and other UOD methods compared in the paper with default settings, simply run the following command:

```bash
python outlier_experiments.py
```

This will automatically run all UOD methods reported in the manuscript.  Please see ```outlier_experiments.py``` for more details.

After training, to print UOD results for a specific algorithm in AUROC/AUPR, run:

```bash
# AUROC of E3Outlier on CIFAR10 with outlier ratio 0.1
python evaluate_roc_auc.py --dataset cifar10 --algo_name e3outlier-0.1

# AUPR of CAE-IF on MNIST with outlier ratio 0.25 and inliers as the postive class
python evaluate_pr_auc.py --dataset mnist --algo_name cae-iforest-0.25 --postive inliers
```

The algorithm names are defined in ```outlier_experiments.py```.

## Citation

```
@incollection{wang_effective_2019,
  title = {Effective End-to-end Unsupervised Outlier Detection via Inlier Priority of Discriminative Network},
  author = {Wang, Siqi and Zeng, Yijie and Liu, Xinwang and Zhu, En and Yin, Jianping and Xu, Chuanfu and Kloft, Marius},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2019}
}
```

## License

E3Outlier is released under the MIT License.
