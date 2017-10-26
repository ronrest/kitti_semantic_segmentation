from __future__ import print_function, division, unicode_literals
import tensorflow as tf
import numpy as np
import pickle
import os
import time
import shutil  # for removing dirs
# import distutils

from data_processing import create_data_dict, str2file, id2label, label2id, pickle2obj, obj2pickle, maybe_make_pardir
from image_processing import create_augmentation_func_for_segmentation
from architectures import arc

import argparse
p = argparse.ArgumentParser()
p.add_argument("name", type=str, help="Model Name")
p.add_argument("--arc", type=str, help="Model Architecture")
p.add_argument("-d", "--data", type=str, default="data", help="Path to directory containing the data")
p.add_argument("--pretrained_snapshot", type=str, default=None, help="Path to pretrained snapshot (if doing transfer learning)")
p.add_argument("-v", "--n_valid",type=int, default=31, help="Num samples to set aside for validation set")
p.add_argument("-m", "--max_data", type=int, default=100000000, help="Max number of samples to use from training data. Useful for quickly testing a training reigeme")
p.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
p.add_argument("-a", "--alpha", type=float, default=0.001, help="Learning rate alpha")
p.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (amount to drop)")
p.add_argument("-n", "--n_epochs", type=int, default=1, help="Number of epochs")
p.add_argument("-p", "--print_every", type=int, default=100, help="How often to print out feedback on training (in number of steps)")
p.add_argument("-l", "--l2", type=float, default=None, help="Amount of L2 to apply")
p.add_argument("-s", "--img_dim", type=int, default=32, help="Size of single dimension of image (assuming square image)")
p.add_argument("--best_metric", type=str, default="valid_iou", help="The metric to use for evaluating best model")
p.add_argument("--aug_func", type=str, default="a", help="Aug function to use [a, b]")
p.add_argument("--dynamic", action='store_true', help="Toggle switch to turn on dynamic loading of data from raw image files")

opt = p.parse_args()


# ##############################################################################
#                               DATA
# ##############################################################################
print(("#"*70)+"\n"+"PREPARING DATA"+"\n"+("#"*70))

MAX_DATA = opt.max_data
N_VALID = opt.n_valid

print("DYNAMIC: ", opt.dynamic)
if opt.dynamic:
    print("DEBUG: using dynamic data")
    data = create_data_dict(datadir=opt.data)
else:
    print("DEBUG: using prepared image arrays data")
    data = pickle2obj(opt.data)

print("Creating validation split")
data["X_valid"] = data["X_train"][:N_VALID]
data["Y_valid"] = data["Y_train"][:N_VALID]
data["X_train"] = data["X_train"][N_VALID:N_VALID+MAX_DATA]
data["Y_train"] = data["Y_train"][N_VALID:N_VALID+MAX_DATA]

# Visualization data
n_viz = 25
data["X_train_viz"] = data["X_train"][:n_viz]
data["Y_train_viz"] = data["Y_train"][:n_viz]


# Information about data shapes
print("DATA SIZES")
print("- X_train: ", len(data["X_train"])) #
print("- Y_train: ", len(data["Y_train"])) #
# print("- X_test : ", len(data["X_test"]))  #
# print("- Y_test : ", len(data["Y_test"]))  #
print("- X_valid: ", len(data["X_valid"])) #
print("- Y_valid: ", len(data["Y_valid"])) #

# ##############################################################################
#                                                              DATA AUGMENTATION
# ##############################################################################
aug_funcA = create_augmentation_func_for_segmentation(
    shadow=(0.01, 0.8), # (0.01, 0.7)
    shadow_file="shadow_pattern.jpg",
    shadow_crop_range=(0.02, 0.5),
    rotate=30, #15,
    crop=0.66,
    lr_flip=True,
    tb_flip=False,
    brightness=(0.5, 0.4, 4),
    contrast=(0.5, 0.3, 5),
    blur=2, # 1
    noise=6 #4
    )


aug_funcB = create_augmentation_func_for_segmentation(
    shadow=(0.5, 0.85), # (0.01, 0.7)
    shadow_file="shadow_pattern.jpg",
    shadow_crop_range=(0.02, 0.5),
    rotate=45, #30,
    crop=0.66,
    lr_flip=True,
    tb_flip=False,
    brightness=(0.5, 0.4, 4),
    contrast=(0.5, 0.3, 5),
    blur=2, # 1
    noise=4 #4
    )

aug_funcs = {}
aug_funcs["a"] = aug_funcA
aug_funcs["b"] = aug_funcB
aug_funcs["None"] = None

# VISUALIZE THE RANDOM TRANSFORMATIONS
# from viz import viz_sample_seg_augmentations
# viz_sample_seg_augmentations(data["X_train"], data["Y_train"], aug_func=aug_func, n_images=5, n_per_image=5, saveto=None).show()

# ##############################################################################
#                                                         CREATE AND TRAIN MODEL
# ##############################################################################
def create_and_train_model(
        name,
        ModelClass,
        data,
        n_classes=1,
        pretrained_snapshot=None,
        dynamic=False,
        alpha=0.01,
        dropout=0.0,
        l2=None,
        n_epochs=30,
        batch_size=128,
        print_every=None,
        overwrite=False,
        img_shape=None,
        augmentation_func=None,
        best_evals_metric="valid_acc",
        viz_every=10,
        ):
    print("\n"+("#"*70)+"\n"+"MODEL NAME = "+name+"\n"+("#"*70)+"\n")
    print("ALPHA: ", alpha)
    print("DROPOUT: ", dropout)
    print("BATCH SIZE: ", batch_size)
    model_dir = os.path.join("models", name)

    # Check if the model already exists
    if os.path.exists(model_dir):
        template = ("="*70) +"\n"+("="*70) +"\n"+(" "*30)+"IMPORTANT!\n"+ ("-"*70)+"\nModel with this name already exists.\n{}\n\n"+("="*70)+"\n"+("="*70)+"\n"
        if overwrite:
            print(template.format("WARNING!!!: YOU ARE IN OVERWRITE MODE\nCompletely deleting the directory associated with the previous model"))
            shutil.rmtree(model_dir)
        else:
            print(template.format("Attempting to re-use existing files"))

    # Create model object
    if dynamic:
        assert img_shape is not None, "Need to feed image shape for dynamic option"
        width, height = img_shape
    else:
        img_shape = list(data["X_train"].shape[1:3])
    # n_classes = len(id2label)

    kwargs = {
        "name":name,
        "img_shape":img_shape,
        "n_channels":3,
        "n_classes":n_classes,
        "dynamic":dynamic,
        "l2":l2,
        "best_evals_metric":best_evals_metric,
        }

    if pretrained_snapshot:
        kwargs["pretrained_snapshot"] = pretrained_snapshot

    model = ModelClass(**kwargs)
    model.create_graph()

    # Train the model
    model.train(data, alpha=alpha, dropout=dropout, n_epochs=n_epochs, batch_size=batch_size, print_every=print_every, augmentation_func=augmentation_func, viz_every=viz_every)
    print("DONE TRAINING")


# ##############################################################################
#                                                                           MAIN
# ##############################################################################
n_classes = 1
create_and_train_model(
        name = opt.name,
        ModelClass = arc[opt.arc],
        data = data,
        n_classes=n_classes,
        pretrained_snapshot = opt.pretrained_snapshot,
        dynamic = opt.dynamic,
        alpha=opt.alpha,
        dropout=opt.dropout,
        l2=opt.l2,
        n_epochs=opt.n_epochs,
        batch_size=opt.batch_size,
        print_every=opt.print_every,
        overwrite=False,
        img_shape=(opt.img_dim, opt.img_dim),
        augmentation_func=aug_funcs[opt.aug_func],
        best_evals_metric=opt.best_metric,
        viz_every=10
        )
