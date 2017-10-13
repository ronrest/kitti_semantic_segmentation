import numpy as np
import matplotlib
matplotlib.use('AGG') # make matplotlib work on aws
import matplotlib.pyplot as plt

# ==============================================================================
#                                                                   TRAIN_CURVES
# ==============================================================================
def train_curves(train, valid, saveto=None, title="Accuracy over time", ylab="accuracy", legend_pos="lower right"):
    """ Plots the training curves. If `saveto` is specified, it saves the
        the plot image to a file instead of showing it.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle(title, fontsize=15)
    ax.plot(train, color="#FF4F40",  label="train")
    ax.plot(valid, color="#307EC7",  label="valid")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylab)

    # Grid lines
    ax.grid(True)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='#888888', linestyle='-')
    plt.grid(b=True, which='minor', color='#AAAAAA', linestyle='-', alpha=0.2)

    # Legend
    ax.legend(loc=legend_pos, title="", frameon=False,  fontsize=8)

    # Save or show
    if saveto is None:
        plt.show()
    else:
        fig.savefig(saveto)
        plt.close()


# ==============================================================================
#                                                                   BATCH 2 GRID
# ==============================================================================
def batch2grid(imgs, rows, cols):
    """
    Given a batch of images stored as a numpy array of shape:

           [n_batch, img_height, img_width]
        or [n_batch, img_height, img_width, n_channels]

    it creates a grid of those images of shape described in `rows` and `cols`.

    Args:
        imgs: (numpy array)
            Shape should be either:
                - [n_batch, im_rows, im_cols]
                - [n_batch, im_rows, im_cols, n_channels]

        rows: (int) How many rows of images to use
        cols: (int) How many cols of images to use

    Returns: (numpy array)
        The grid of images as one large image of either shape:
            - [n_classes*im_cols, num_per_class*im_rows]
            - [n_classes*im_cols, num_per_class*im_rows, n_channels]
    """
    # TODO: have a resize option to rescale the individual sample images
    # TODO: Have a random shuffle option
    # TODO: Set the random seed if needed
    # if seed is not None:
    #     np.random.seed(seed=seed)

    # Only use the number of images needed to fill grid
    assert rows>0 and cols>0, "rows and cols must be positive integers"
    n_cells = (rows*cols)
    imgs = imgs[:n_cells]

    # Image dimensions
    n_dims = imgs.ndim
    assert n_dims==3 or n_dims==4, "Incorrect # of dimensions for input array"

    # Deal with images that have no color channel
    if n_dims == 3:
        imgs = np.expand_dims(imgs, axis=3)

    n_batch, img_height, img_width, n_channels = imgs.shape

    # Handle case where there is not enough images in batch to fill grid
    n_gap = n_cells - n_batch
    imgs = np.pad(imgs, pad_width=[(0,n_gap),(0,0), (0,0), (0,0)], mode="constant", constant_values=0)

    # Reshape into grid
    grid = imgs.reshape(rows,cols,img_height,img_width,n_channels).swapaxes(1,2)
    grid = grid.reshape(rows*img_height,cols*img_width,n_channels)

    # If input was flat images with no color channels, then flatten the output
    if n_dims == 3:
        grid = grid.squeeze(axis=2) # axis 2 because batch dim has been removed

    return grid

# ==============================================================================
#                                                                         VIZSEG
# ==============================================================================
import PIL
from PIL import Image, ImageChops
from data_processing import maybe_make_pardir
def vizseg(img, label, pred=None, saveto=None):
    """ Given an input image, the segmentation labels for the pixels,
        and, OPTIONALLY, separate segmentation predictions, It returns
        an image that overlays the predictions on top of the original
        image color coded as:

        blue    = ground truth label
        red     = prediction
        magenta = where ground truth overlaps with prediction
    Args:
        img:    (numpy array) of shape [height, width, 3] of int vals 0-255
        label:  (numpy array) of shape [height, width] of int vals 0-n_classes
        pred:   (numpy array) of shape [height, width] of int vals 0-n_classes
    """
    assert 2 == label.ndim == (pred.ndim if pred is not None else 2), \
        "Label and Prediction MUST be of shape 2D arrays with no color channel or batch axis"
    assert (img.ndim == 3) and (img.shape[-1] == 3), \
        "Input image should be of shape [n_rows, n_cols, 3]"
    assert img.shape[:2] == label.shape == (pred.shape if pred is not None else label.shape), \
        "Image height and width for img, label, and pred must match up"


    # Convert chanel axis to one hot encodings (max of three classes for 3 chanels)
    label = np.eye(3, dtype=np.uint)[label]
    if pred is not None:
        pred = np.eye(3, dtype=np.uint8)[pred]

    # Extract JUST the road class (class 1)
    # Red for prediction, Blue for label
    road = np.zeros_like(label, dtype=np.uint8)
    road[:,:,2] = label[:,:,1]*255
    if pred is not None:
        road[:,:,0] = pred[:,:,1]*255

    # Overlay the input image with the label and prediction
    img = PIL.Image.fromarray(img)
    road = PIL.Image.fromarray(road)
    overlay = PIL.ImageChops.add(img, road, scale=1.5)

    if saveto is not None:
        maybe_make_pardir(saveto)
        overlay.save(saveto, "JPEG")

    return overlay


# ==============================================================================
#                                                   VIZ_SAMPLE_SEG_AUGMENTATIONS
# ==============================================================================
def viz_sample_seg_augmentations(X, Y, aug_func, n_images=5, n_per_image=5, saveto=None):
    """ Given a batch of data X, and Y,  it takes n_images samples, and performs
        `n_per_image` random transformations for segmentation data on each of
        those images. It then puts them in a grid to visualize. Grid size is:
            n_images wide x n_per_image high

    Args:
        X:          (np array) batch of images
        Y:          (np array) batch of labels images
        aug_func:   (func) function with API `aug_func(X, Y)` that performs
                    random transformations on the images for segmentation
                    purposes.
        n_images:   (int)
        n_per_image:(int)
        saveto:     (str or None)

    Returns: (None, or PIL image)
        If saveto is provided, then it saves teh image and returns None.
        Else, it returns the PIL image.
    Examples:
        samples = viz_sample_seg_augmentations(data["X_train"], data["Y_train"],
            aug_func=aug_func, n_images=5, n_per_image=5, saveto=None)
        samples.show()
    """
    X = X[:n_images]
    Y = Y[:n_images]
    gx = []
    gy = []

    # Perform Augmentations
    for col in range(n_per_image):
        x, y = aug_func(X, Y)
        gx.append(x)
        gy.append(y)

    # Put into a grid
    _, height, width, n_channels = X.shape
    gx = np.array(gx, dtype=np.uint8).reshape(n_images*n_per_image, height, width, n_channels)
    gy = np.array(gy, dtype=np.uint8).reshape(n_images*n_per_image, height, width)
    gx = batch2grid(gx, n_images, n_per_image)
    gy = batch2grid(gy, n_images, n_per_image)

    # Overlay labels on top of image
    return vizseg(gx, gy, saveto=saveto)
