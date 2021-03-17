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

Synthetic noise type:

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

[1] Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., ... & Sugiyama, M. (2018, January). Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels. In 32nd Conference on Neural Information Processing Systems (NIPS). NEURAL INFORMATION PROCESSING SYSTEMS (NIPS).

[2] Yu, X., Han, B., Yao, J., Niu, G., Tsang, I., & Sugiyama, M. (2019, May). How does disagreement help generalization against label corruption?. In International Conference on Machine Learning (pp. 7164-7173). PMLR.

[3] Malach, E., & Shalev-Shwartz, S. (2017, January). Decoupling" when to update" from" how to update". In 31nd Conference on Neural Information Processing Systems (NIPS). NEURAL INFORMATION PROCESSING SYSTEMS (NIPS).

[4] Wei, H., Feng, L., Chen, X., & An, B. (2020). Combating noisy labels by agreement: A joint training method with co-regularization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 13726-13735).