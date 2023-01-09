# limit the number of cpus used by high performance libraries
from ctypes import resize
import os
from pickle import NONE

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys

import logging

import argparse


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import tensorflow as tf


# Import for FairMOT
sys.path.insert(0, "./fair_mot/lib")
import fair_mot.lib.datasets.dataset.jde as datasets
from fair_mot.lib.tracker.multitracker import JDETracker


"""Import for pose estimation"""
sys.path.insert(0, "./pose_estimation")
from pose_estimation.tf_pose.estimator import TfPoseEstimator
from pose_estimation.tf_pose.networks import get_graph_path, model_wh
from pose_estimation.tf_pose.estimator import Human


# Import for PCPA
from action_predict import action_prediction
from utils_PCPA import *


# Import for MASK_PCPA
# Import for local_context_cnn
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Import for deeplabv3
from action_predict import DeepLabModel
from action_predict import create_cityscapes_label_colormap
from action_predict import label_to_color_image
from action_predict import init_canvas
from PIL import Image

# Import for RAFT
sys.path.append("./raft_core")
from raft_core.raft import RAFT
from raft_core.utils.utils import InputPadder
from raft_core.utils.flow_viz import flow_to_image

# Import for speed
from efficientnet_pytorch import EfficientNet
from torchsummary import summary

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class Predict:
    def __init__(self, opt):
        self.model_pose = TfPoseEstimator(
            get_graph_path("egen_jaad_1_5"), target_size=(100, 100)
        )

        self.model_dn = self.load_densenet(opt.dense_model)  # Load densenet model

        # Tracker
        self.dataloader = datasets.LoadVideo(opt.source, (1088, 608))
        self.tracker = JDETracker(opt, frame_rate=self.dataloader.frame_rate)

        base_model = VGG19(weights="imagenet")
        self.VGGmodel = Model(
            inputs=base_model.input, outputs=base_model.get_layer("block4_pool").output
        )
        # backbone_dict = {"vgg16": vgg16.VGG16, "resnet50": resnet50.ResNet50}

        # load segmentation model
        segmodel_path = "deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz"
        self.segmodel = DeepLabModel(segmodel_path)
        
        # load optical flow model
        optmodel_path = "./raft_core/models/raft-things.pth"
        model = torch.nn.DataParallel(RAFT(opt))
        model.load_state_dict(torch.load(optmodel_path))
        model = model.module
        model.to(opt.device)
        model.eval() 
        self.model_opt = model
        
        # load speed model
        speedmodel_path = "./speed_model/95.pt"
        V = 0     # what version of efficientnet did you use
        IN_C = 2  # number of input channels
        NUM_C = 1 # number of classes to predict
        self.model_spd = torch.load(speedmodel_path)
        self.model_spd.to(opt.device)

    def load_densenet(self, model_path):
        model_dn = tf.keras.models.load_model(model_path)
        model_dn.summary()
        return model_dn

    def intention_pred(self, X_test):
        predictions = self.model_dn.predict(X_test, verbose=1)
        Y = np.round(predictions[0][0])
        return Y

    def tracker_update(self, blob, im0s):
        return self.tracker.update(blob, im0s)

    def pose_inference(self, cropped, resize_to_default, upsample_size):
        return self.model_pose.inference(cropped, resize_to_default, upsample_size)

    def optical_inference(self, image1, image2):
        image1 = torch.from_numpy(image1).float()[None].to(opt.device)
        image2 = torch.from_numpy(image2).float()[None].to(opt.device)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        _, flow_up = self.model_opt(image1, image2, iters=20, test_mode=True)
        return flow_up
    
    
    def optical_vis(self, of, output_dir):
        output_dir = Path(output_dir)
  
        of = of[0].permute(1,2,0).numpy()
        of = flow_to_image(of)
        img = Image.fromarray(of)
        img.save(output_dir)
        
        
                
    
    def speed_inference(self, image1, image2, frame_idx, source):
        
        print(f'Shape of Image_1: {image1.shape}')
        
        # image1 = cv2.resize(np.transpose(image1, (1, 2, 0)), (640, 480))
        # image2 = cv2.resize(np.transpose(image2, (1, 2, 0)), (640, 480))
        
        
        image1 = cv2.resize(image1, (960, 540))
        image2 = cv2.resize(image2, (960, 540))
        
        
        print(f'Shape of Image_1: {image1.shape}')
        
        image1 = np.transpose(image1, (2, 0, 1))
        image2 = np.transpose(image2, (2, 0, 1))
        
        print(f'Shape of Image_1: {image1.shape}')
        
        
        optical_flow = self.optical_inference(image1, image2)
        pred = self.model_spd(optical_flow)
        

        # optical_flow = optical_flow.cpu()
        # path = os.path.join('optical_images', str(source), str(frame_idx) + '.jpg')
        # os.makedirs(os.path.join('optical_images', str(source)), exist_ok=True)
        # self.optical_vis(optical_flow, path)
        
        
        del optical_flow
        torch.cuda.empty_cache()
        return pred.item()




def detect(opt):
    (
        source,
        save_vid,
        save_txt,
        project,
        name,
        exist_ok,
        dense_model,
        eratio,
        target_dim,
        speed_file,
    ) = (
        opt.source,
        opt.save_vid,
        opt.save_txt,
        opt.project,
        opt.name,
        opt.exist_ok,
        opt.dense_model,
        opt.eratio,
        opt.target_dim,
        opt.speed_file,
    )

    opt.device = select_device(opt.device)

    # Directories

    # exp_name = "exp"
    # save_dir = increment_path(
    #     Path(project) / exp_name, exist_ok=exist_ok
    # )  # increment run if project name exists
    save_dir = Path(project)
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    predict = Predict(opt)

    # Set Dataloader
    vid_path, vid_writer = None, None

    # Dataloader
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split("/")[-1].split(".")[0]
    txt_path = str(Path(save_dir)) + "/" + txt_file_name + ".txt"

    rolling_data = {}
    results = []
    resize_out_ratio = 4.0
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    prev_img = None
   
    speeds = []
    with open(speed_file, "r") as f: 
        speeds = f.readlines()

    for frame_idx, (vid_cap, path, img, im0s) in enumerate(predict.dataloader):
        seen += 1
        p = Path(path)  # to Path
        save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
        im0 = im0s.copy()

        # predict vehicle speed
        vehicle_speed = float(speeds[frame_idx])
        # vehicle_speed = 0
        # if prev_img is not None:
        #     vehicle_speed = predict.speed_inference(prev_img, im0s.copy(), frame_idx, source)
        print("!!!!!!!!!!!!!!!!!\n")
        print("speed:", vehicle_speed)
        print("!!!!!!!!!!!!!!!!!\n")
        # prev_img = im0s.copy()
        
        # run tracking
        if opt.device.type != "cpu":
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)

        t0 = time_sync()
        online_targets = predict.tracker_update(blob, im0s)
        peds_with_ID = []

        for t in online_targets:
            x, y, w, h = t.tlwh
            tid = t.track_id
            vertical = w / h > 1.6
            outside = x < 0 or y < 0 or x + w >= im0s.shape[1] or y + h >= im0s.shape[0]
            if w * h > opt.min_box_area and not vertical and not outside:
                peds_with_ID.append((t.tlwh, tid, t.score))

        t1 = time_sync()
        dt[0] = t1 - t0

        im0 = cv2.putText(
                im0,
                "Speed: %d" % (vehicle_speed),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )


        for xywh, p_id, conf in peds_with_ID:
            x, y, w, h = (int(n) for n in xywh)
            black_skelett = []

            # Local box contain pedestrian and cordinate of bounding box
            bbox = []
            pose_coordinates = []
            local_context_features = []
            global_context_features = []

            # plot the skeletons
            try:
                cropped = im0s[y : y + h, x : x + w].copy()

                bbox = [x, y, x + w, y + h]

                img_data = im0s.copy()

                if "MASK" in dense_model:
                    local_context_features = load_local_context_feature(
                        im0s.copy(), bbox, eratio, True, predict.VGGmodel, (224, 224)
                    )
                    global_context_features = load_global_context_feature(
                        im0s.copy(),
                        bbox,
                        predict.VGGmodel,
                        predict.segmodel,
                        (224, 224),
                    )
                else:
                    local_context_features = load_local_context_feature(
                        im0s.copy(), bbox, eratio, False, None, (112, 112)
                    )

                humans = predict.pose_inference(
                    cropped,
                    resize_to_default=(w > 0 and h > 0),
                    upsample_size=resize_out_ratio,
                )
                humans.sort(key=lambda human: human.score, reverse=True)

                print("\n\nPedestrian ID: ", p_id)
                print("Body_parts: ", len(humans))

                if len(humans) != 0:

                    print("Length of pose key: ", len(humans[0].body_parts.keys()))
                    print("Keys of pose: ", humans[0].body_parts.keys())

                    for i in range(18):

                        if i in humans[0].body_parts.keys():

                            body_part = humans[0].body_parts[i]
                            body_part_x, body_part_y = x + int(
                                body_part.x * w + 0.5
                            ), y + int(body_part.y * h + 0.5)
                            pose_coordinates.append(body_part_x)
                            pose_coordinates.append(body_part_y)
                        else:
                            pose_coordinates.append(x)
                            pose_coordinates.append(y)
                else:
                    pose_coordinates += [0] * 36
                print("Length of coordinates: ", len(pose_coordinates))
                # im0s2 = im0s.copy()

            except Exception as e:
                print(e)
                # im0s2 = im0s.copy()

            # looking for previous 16 frames data for a given pedestrian:

            intent = 0  # (default, the pedestrian is not crossing)

            if p_id in rolling_data:

                if len(rolling_data[p_id][0]) == 16:

                    seq = rolling_data[p_id].copy()

                    for i in range(0, len(seq)):
                        seq[i] = np.array(seq[i])
                        seq[i] = np.expand_dims(seq[i], axis=0)
                        print("Type of seq: ", type(seq))
                        print("Type of seq[i]: ", i, ": ", type(seq[i]))
                        print("Shape of seq( == 16) ", i, ": ", seq[i].shape)

                    intent = predict.intention_pred(seq)

                else:

                    seq = [[x[-1]] * 16 for x in rolling_data[p_id]]
                    for i in range(0, len(seq)):
                        seq[i] = np.array(seq[i])
                        seq[i] = np.expand_dims(seq[i], axis=0)
                        print("Type of seq: ", type(seq))
                        print("Type of seq: ", i, ": ", type(seq[i]))
                        print("Shape of seq( < 16) ", i, ": ", seq[i].shape)

                    intent = predict.intention_pred(seq)

            # risky pedestrian identification thru box color

            if intent == 1:
                color = (0, 0, 255)  # Red -> Crossing
            else:
                color = (0, 255, 0)  # Green -> Not crossing

            im0 = cv2.rectangle(
                im0, (int(x), int(y)), (int(x + w), int(y + h)), color, 2
            )
            im0 = cv2.putText(
                im0,
                str(p_id) + " " + str(round(float(conf), 2)),
                (x, y - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 0, 0),
                thickness=2,
            )
            im0 = cv2.putText(
                im0,
                "Frame No: %d" % (frame_idx),
                (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

            results.append([frame_idx, x, y, w + x, h + y, intent])
            print(frame_idx, p_id, x, y, w + x, h + y, intent)

            # storing the data for last 16 frames
            try:

                if "MASK" in dense_model:
                    if p_id in rolling_data:  # ID exists in dict

                        if (
                            len(rolling_data[p_id][0]) < 16
                        ):  # bboxes values for 16 frames

                            rolling_data[p_id][0].append(local_context_features)
                            rolling_data[p_id][1].append(global_context_features)
                            rolling_data[p_id][2].append(pose_coordinates)
                            rolling_data[p_id][3].append(bbox)
                            rolling_data[p_id][4].append(vehicle_speed)

                        else:

                            for data in rolling_data[p_id]:
                                del data[0]

                            rolling_data[p_id][0].append(local_context_features)
                            rolling_data[p_id][1].append(global_context_features)
                            rolling_data[p_id][2].append(pose_coordinates)
                            rolling_data[p_id][3].append(bbox)
                            rolling_data[p_id][4].append(vehicle_speed)

                    else:
                        rolling_data[p_id] = [[], [], [], [], []]

                        rolling_data[p_id][0].append(local_context_features)
                        rolling_data[p_id][1].append(global_context_features)
                        rolling_data[p_id][2].append(pose_coordinates)
                        rolling_data[p_id][3].append(bbox)
                        rolling_data[p_id][4].append(vehicle_speed)
                else:
                    if p_id in rolling_data:  # ID exists in dict

                        if (
                            len(rolling_data[p_id][0]) < 16
                        ):  # bboxes values for 16 frames

                            rolling_data[p_id][0].append(local_context_features)
                            rolling_data[p_id][1].append(pose_coordinates)
                            rolling_data[p_id][2].append(bbox)

                        else:

                            for data in rolling_data[p_id]:
                                del data[0]

                            rolling_data[p_id][0].append(local_context_features)
                            rolling_data[p_id][1].append(pose_coordinates)
                            rolling_data[p_id][2].append(bbox)

                    else:
                        rolling_data[p_id] = [[], [], []]

                        rolling_data[p_id][0].append(local_context_features)
                        rolling_data[p_id][1].append(pose_coordinates)
                        rolling_data[p_id][2].append(bbox)

            except:
                pass

        # Save results (image with detections)
        if save_vid:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    # print(f'FPS: {fps}')
                    # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    # print(f'Width: {w}, Height: {h}\n')
                    w, h = im0.shape[1], im0.shape[0]
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]

                # print(f'Width: {w}, Height: {h}\n')
                vid_writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            vid_writer.write(im0)

    # save bbox and intention as .txt file
    if save_txt:
        with open(txt_path, "w") as f:
            for result in results:
                for e in result:
                    f.write(str(e) + " ")
                f.write("\n")

    if seen == 0:
        print("seen = 0")
    else:
        t = tuple(x / seen * 1e3 for x in dt)
        logging.info(f"Speed: {t[0]}ms FairMOT\n")

    if save_txt or save_vid:
        print("Results saved to %s" % save_path)
        if platform == "darwin":  # MacOS
            os.system("open " + save_path)


def load_local_context_feature(
    img, bbox, eratio, mask_pcpa, VGGmodel, target_dim=(224, 224)
):

    if not mask_pcpa:
        bbox = list(map(int, bbox[0:4]))
        cropped_image = img[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
        local_context_features = img_pad(
            cropped_image, mode="pad_resize", size=target_dim[0]
        )
    else:
        bbox = list(map(int, bbox[0:4]))
        cropped_image = img[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
        img = img_pad(cropped_image, mode="pad_resize", size=target_dim[0])

        # img = img_pad(img, mode="pad_resize", size=target_dim[0])
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.vgg19.preprocess_input(x)
        block4_pool_features = VGGmodel.predict(x)
        local_context_features = block4_pool_features
        local_context_features = tf.nn.avg_pool2d(
            local_context_features,
            ksize=[14, 14],
            strides=[1, 1, 1, 1],
            padding="VALID",
        )
        local_context_features = tf.squeeze(local_context_features)
        local_context_features = local_context_features.numpy()

    return local_context_features


def load_global_context_feature(
    img_data, bbox, VGGmodel, segmodel, target_dim=(224, 224)
):
    ori_dim = img_data.shape
    bbox = list(map(int, bbox[0:4]))
    ## img_data --- > mask_img_data (deeplabV3)
    original_im = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
    resized_im, seg_map = segmodel.run(original_im)
    resized_im = np.array(resized_im)
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
    seg_image = cv2.addWeighted(resized_im, 0.5, seg_image, 0.5, 0)
    img_data = cv2.resize(seg_image, (ori_dim[1], ori_dim[0]))

    ped_mask = init_canvas(bbox[2] - bbox[0], bbox[3] - bbox[1], color=(255, 255, 255))
    img_data[bbox[1] : bbox[3], bbox[0] : bbox[2]] = ped_mask

    # cv2.imwrite("Mask_image2.jpg", img_data)

    img_features = cv2.resize(img_data, target_dim)
    img = Image.fromarray(cv2.cvtColor(img_features, cv2.COLOR_BGR2RGB))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.vgg19.preprocess_input(x)
    block4_pool_features = VGGmodel.predict(x)
    img_features = block4_pool_features
    img_features = tf.nn.avg_pool2d(
        img_features, ksize=[14, 14], strides=[1, 1, 1, 1], padding="VALID"
    )
    img_features = tf.squeeze(img_features)
    # with tf.compact.v1.Session():
    img_features = img_features.numpy()

    return img_features


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def select_device(device="", batch_size=0, newline=True):
    # device = 'cpu' or '0' or '0,1,2,3'
    # device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    if cpu:
        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        # os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    cuda = not cpu and torch.cuda.is_available()

    return torch.device("cuda:0" if cuda else "cpu")


def time_sync():
    # Pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, default="0", help="source")  # file/folder

    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--save-vid", action="store_true", help="save video tracking results"
    )
    parser.add_argument(
        "--save-txt", action="store_true", help="save MOT compliant results to *.txt"
    )
    parser.add_argument(
        "--project", default=ROOT / "runs/track", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )

    parser.add_argument(
        "--dense_model",
        type=str,
        help="PCPA Model to use",
        required=True,
    )

    # FairMOT
    parser.add_argument(
        "--min-box-area", type=float, default=100, help="filter out tiny boxes"
    )

    parser.add_argument(
        "--arch",
        default="dla_34",
        help="model architecture. Currently tested"
        "resdcn_34 | resdcn_50 | resfpndcn_34 |"
        "dla_34 | hrnet_18",
    )
    parser.add_argument(
        "--head_conv",
        type=int,
        default=-1,
        help="conv layer channels for output head"
        "0 for no conv layer"
        "-1 for default setting: "
        "256 for resnets and 256 for dla.",
    )
    parser.add_argument(
        "--not_reg_offset", action="store_true", help="not regress local offset."
    )

    parser.add_argument(
        "--load_model",
        default="./fair_mot/models/fairmot_dla34.pth",
        help="path to pretrained model",
    )
    parser.add_argument(
        "--conf_thres", type=float, default=0.4, help="confidence thresh for tracking"
    )
    parser.add_argument("--track_buffer", type=int, default=30, help="tracking buffer")
    parser.add_argument(
        "--K", type=int, default=500, help="max number of output objects."
    )
    parser.add_argument(
        "--ltrb", default=True, help="regress left, top, right, bottom of bbox"
    )
    parser.add_argument(
        "--reid_dim", type=int, default=128, help="feature dim for reid"
    )
    parser.add_argument(
        "--down_ratio",
        type=int,
        default=4,
        help="output stride. Currently only supports 4.",
    )

    # FairPCPA
    parser.add_argument(
        "--eratio", type=float, default=1.5, help="enlarge ratio of local_context"
    )

    parser.add_argument(
        "--target_dim", type=int, default=112, help="size of input image feature"
    )

    parser.add_argument(
        "--speed_file", type=str, help="speed file"
    )
    
    opt = parser.parse_args()

    opt.num_classes = 1

    opt.heads = {
        "hm": opt.num_classes,
        "wh": 2 if not opt.ltrb else 4,
        "id": opt.reid_dim,
    }

    if opt.head_conv == -1:  # init default head_conv
        opt.head_conv = 256 if "dla" in opt.arch else 256

    opt.reg_offset = not opt.not_reg_offset
    if opt.reg_offset:
        opt.heads.update({"reg": 2})

    return opt


if __name__ == "__main__":

    opt = parse_args()

    with torch.no_grad():
        detect(opt)
