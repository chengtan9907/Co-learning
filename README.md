# Co-training-based_noisy-label-learning methods

A unified framework for co-training-based noisy label learning methods.

# Introduction

Algorithms:

- [x] [Decoupling](algorithms/Decoupling.py)
- [x] [Co-teaching](algorithms/Coteaching.py)
- [x] [Co-teaching+](algorithms/Coteachingplus.py)
- [x] [JoCoR](algorithms/JoCoR.py)

Datasets:

- [x] [CIFAR-10]
- [x] [CIFAR-100]

Synthetic noise types:

- [x] 'sym'  (Symmetric noisy labels)
- [x] 'asym' (Asymmetric noisy labels)
- [x] 'ins'  (Instance-dependent noisy labels)

# Dependency

1. numpy
2. torch, torchvision
3. scipy
4. addict
5. matplotlib

# Reference

[1] Malach, Eran, et al. "Decoupling" when to update" from" how to update"." NeurIPS 2017.

[2] Han, Bo, et al. "Co-teaching: Robust training of deep neural networks with extremely noisy labels." NeurIPS 2018.

[3] Yu, Xingrui, et al. "How does disagreement help generalization against label corruption?." ICML 2019.

[4] Wei, Hongxin, et al. "Combating noisy labels by agreement: A joint training method with co-regularization." CVPR 2020.

[5] Ma, Xingjun, et al. "Normalized loss functions for deep learning with noisy labels." ICML 2020.