               Bilinear CNN (B-CNN) for Fine-grained recognition


DESCRIPTIONS
    After getting the deep descriptors of an image, bilinear pooling computes
    the sum of the outer product of those deep descriptors. Bilinear pooling
    captures all pairwise descriptor interactions, i.e., interactions of
    different part, in a translational invariant manner.

    B-CNN provides richer representations than linear models, and B-CNN achieves
    better performance than part-based fine-grained models with no need for
    further part annotation.
    
    Please note that this repo is relative old, which is writen in PyTorch 0.3.0.
    If you are using newer version of PyTorch (say, >=0.4.0), it is suggested to
    consider using this repo https://github.com/HaoMood/blinear-cnn-faster instead.


REFERENCE
    T.-Y. Lin, A. RoyChowdhury, and S. Maji. Bilinear CNN models for
    fine-grained visual recognition. In Proceedings of the IEEE International
    Conference on Computer Vision, pages 1449--1457, 2015.


PREREQUIREMENTS
    Python3.6 with Numpy supported
    PyTorch


LAYOUT
    ./data/                 # Datasets
    ./doc/                  # Automatically generated documents
    ./src/                  # Source code


USAGE
    Step 1. Fine-tune the fc layer only. It gives 76.77% test set accuracy.
    $ CUDA_VISIBLE_DEVICES=0,1,2,3 ./src/bilinear_cnn_fc.py --base_lr 1.0 \
          --batch_size 64 --epochs 55 --weight_decay 1e-8 \
          | tee "[fc-] base_lr_1.0-weight_decay_1e-8-epoch_.log"

    Step 2. Fine-tune all layers. It gives 84.17% test set accuracy.
    $ CUDA_VISIBLE_DEVICES=0,1,2,3 ./src/bilinear_cnn_all.py --base_lr 1e-2 \
          --batch_size 64 --epochs 25 --weight_decay 1e-5 \
          --model "model.pth" \
          | tee "[all-] base_lr_1e-2-weight_decay_1e-5-epoch_.log"


AUTHOR
    Hao Zhang: zhangh0214@gmail.com


LICENSE
    CC BY-SA 3.0
