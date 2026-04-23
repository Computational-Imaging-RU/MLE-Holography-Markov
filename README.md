# Maximum Likelihood Reconstruction for Multi-Look Digital Holography with Markov-Modeled Speckle Correlation

## Preview
#### There are 4 python files in this repo.

- train_PGD_MC_complex.py: training PGD-MC algorithm for recovering speckle-free real-valued reflectivity image from complex-valued holographic measurements with Markov modeled multi-look correlated speckle.

- PGD_MC_complex_tensor.py: Efficient implementation of Monte-Carlo sampling and conjugate gradient methods for matrix-free maximum likelihood based reconstruction for digital holography with correlated speckle across looks.

- decoder.py: basic network structures of the Deep Decoder we use for projection.

- utils.py: all the other helper functions.

/data/: test images for evaluation.

## Run the simulation

#### Run the PGD-MC algorithm (efficient Monte-Carlo and conjugate gradient methods) for recovering images from holographic measurements with correlated speckle across looks:

#### E.g., recover images from measurements with number of looks L=4, speckle correlation coefficient alpha=0.8 circular aperture radius ratio=1.0, additive noise level=25, Monte-Carlo samples=50, denoiser=DIP:

```
python train_PGD_MC_complex.py --dataset 'Set11_peppers' --mask_rate 1.0 --alpha 0.8 --num_look 4 --add_std 0.1 --add_std_prime 0.1 --lr_GD 0.01 --outer_ite 100 --num_ite_MC 50 --denoiser 'DIP' --lr_NN 1e-3 --inner_ite 1000
```

## Relevant works on image reconstruction in coherent imaging with speckle

[1] Chen, Xi, Arian Maleki, and Shirin Jalali. "Maximum Likelihood Reconstruction for Multi-Look Digital Holography with Markov-Modeled Speckle Correlation." arXiv preprint arXiv:2604.20154 (2026) [paper](https://arxiv.org/abs/2604.20154)

[2] Chen, Xi, Arian Maleki, and Shirin Jalali. "Monte Carlo Maximum Likelihood Reconstruction for Digital Holography with Speckle." arXiv preprint arXiv:2602.10344 (2026) [paper](https://arxiv.org/pdf/2602.10344)

[3] Chen, Xi, Soham Jana, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Multilook Coherent Imaging: Theoretical Guarantees and Algorithms." arXiv preprint arXiv:2505.23594 (2025) [paper](https://arxiv.org/pdf/2505.23594)

[4] Chen, Xi, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Chen, Xi, et al. "Efficient multilook coherent imaging with temporally dependent speckle noise." Unconventional Imaging, Sensing, and Adaptive Optics 2025. Vol. 13619. SPIE, 2025. [paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13619/1361915/Efficient-multilook-coherent-imaging-with-temporally-dependent-speckle-noise/10.1117/12.3063994.full)

[5] Chen, Xi, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Monte-Carlo Based Efficient Image Reconstruction in Coherent Imaging With Speckle Noise." 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI). IEEE, 2025. [paper](https://ieeexplore.ieee.org/abstract/document/10981291)

[6] Chen, Xi, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Novel approach to coherent imaging in the presence of speckle noise." Unconventional Imaging, Sensing, and Adaptive Optics 2024. Vol. 13149. SPIE, 2024. [paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13149/1314908/Novel-approach-to-coherent-imaging-in-the-presence-of-speckle/10.1117/12.3027824.full)

[7] Chen, Xi, Zhewen Hou, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Bagged Deep Image Prior for Recovering Images in the Presence of Speckle Noise." Forty-first International Conference on Machine Learning (ICML 2024). [paper](https://openreview.net/pdf?id=IoUOhnCmlX)

[8] Chen, Xi, Zhewen Hou, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Multilook compressive sensing in the presence of speckle noise." In NeurIPS 2023 Workshop on Deep Learning and Inverse Problems. 2023. [paper](https://openreview.net/forum?id=G8wMnihF6E)

[9] Zhou, Wenda, Shirin Jalali, and Arian Maleki. "Compressed sensing in the presence of speckle noise." IEEE Transactions on Information Theory 68.10 (2022): 6964-6980. [paper](https://ieeexplore.ieee.org/abstract/document/9783054)
