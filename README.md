# AAL-UA
The implementation of our Additive Adversarial Learning for Unbiased Authentication method by Keras

## Dependencies
The code runs with Python and requires Tensorflow of version 1.2.1 or higher and Keras of version 2.0 or higher. Please `pip install` the following packages:
- `numpy`
- `tensorflow` 
- `keras`
- `pandas`
 
## CMNIST data

Download `colored_mnist.h5` in https://drive.google.com/drive/folders/1EH9GM9TTsfcWxYV5QtwCtyTtJWrcP3h5

And put `colored_mnist.h5` in the `data/` folder

For Stage 1, run the following commands in shell:

```shell
python train_cmnist_phase1.py
```

For Stage 2, copy the name of any resulted h5 file, e.g., `_colored_mnist_data_phase1_AUC_0.9573.h5`, to set value for `read_path` in line 269 of `train_phase2.py`, then run the following commands in shell:

```shell
python train_phase2.py
```

See `train_cmnist_phase1.py` and `train_phase2.py` for details. 

## CelebA data

Download `celeba_img_align_5p_size64.h5` in https://drive.google.com/drive/folders/1EH9GM9TTsfcWxYV5QtwCtyTtJWrcP3h5

And put `celeba_img_align_5p_size64.h5` in the `data/` folder

For Stage 1, run the following commands in shell:

```shell
python train_celeba_phase1.py
```

For Stage 2, similarly as for CMNIST, copy the name of any resulted h5 file to set value for `read_path` in line 269 of `train_phase2.py`, then run the following commands in shell:

```shell
python train_phase2.py
```

See `train_celeba_phase1.py` and `train_phase2.py` for details. 

## Mobile data

Download `device_transfer.h5` in https://drive.google.com/drive/folders/1EH9GM9TTsfcWxYV5QtwCtyTtJWrcP3h5

And put `device_transfer.h5` in the `data/` folder

For Stage 1, run the following commands in shell:

```shell
python train_mobile_phase1.py
```

For Stage 2, similarly as for CMNIST, copy the name of any resulted h5 file to set value for `read_path` in line 269 of `train_phase2.py`, then run the following commands in shell:

```shell
python train_phase2.py
```

See `train_mobile_phase1.py` and `train_phase2.py` for details. 
