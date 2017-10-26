from __future__ import print_function, division
import os
import distutils
import numpy as np
import cv2
from architectures import arc

import argparse
p = argparse.ArgumentParser(description="Run a prediction on video")
p.add_argument("name", type=str, help="model name")
p.add_argument("--arc", type=str, help="model architecture")
p.add_argument("-d", "--img_dim", type=int, default=299, help="image dimension (64, 128, 224, 299)")
p.add_argument("-v", "--vid", type=str, help="Path to the video file to use")
p.add_argument("-s", "--saveto", type=str, default="", help="path to save output video to")
p.add_argument("--best", default=True, type=lambda x:bool(distutils.util.strtobool(x)), help="Use best snapshot? False uses latest snapshot")
# TODO: Add a `show` argument.
opt = p.parse_args()


SHOW_VID = False

# input_dims = (128,  128)
input_dims = (opt.img_dim,  opt.img_dim)

################################################################################
#                                                            START VIDEO STREAM
################################################################################
# FILE PATHS
video_path = opt.vid
print("Using video", video_path)
assert os.path.exists(video_path), "VIDEO DOES NOT EXIST\n- "+video_path

#images = os.listdir(images_dir)

# FOR SAVING VIDEO
out_video_path = opt.saveto.strip()
if out_video_path != "":
    # taken from: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid_dims = (640, 360)# (1280, 720)
    out = cv2.VideoWriter(out_video_path,fourcc, 24.0, out_vid_dims)


################################################################################
#                                                            START VIDEO STREAM
################################################################################
ModelClass = arc[opt.arc]
model = ModelClass(name=opt.name, pretrained_snapshot=None, img_shape=[opt.img_dim, opt.img_dim], n_channels=3, n_classes=1, dynamic=False, best_evals_metric="valid_iou")
model.create_graph()

with model.create_session() as session:
    model.initialize_vars(session=session, best=opt.best)
    # START CAPTURE
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, image = cap.read()
        if ret == True:
            # scaled = cv2.resize(image, (96,  32))
            scaled = cv2.resize(image, input_dims)
            if image is None:
                print("NO IMAGE!!!!")


            height, width, _ = image.shape
            pred = np.zeros_like(scaled)
            pred[:,:,2] = model.predict(np.expand_dims(scaled, axis=0), verbose=False, session=session) == 1
            pred = pred * 255

            pred = cv2.resize(pred, (width,  height))
            overlayed = cv2.addWeighted(image,0.7,pred,0.5,0)
            if SHOW_VID:
                cv2.imshow('Stack', overlayed)

            # SAVE THE FRAME TO VIDEO FILE
            if out_video_path != "":
                outvid = cv2.resize(overlayed, out_vid_dims)
                out.write(outvid)

            # Quit if Escape button pressed
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        else:
            break

    cap.release()
    if out_video_path != "":
        out.release()
    cv2.destroyAllWindows()



