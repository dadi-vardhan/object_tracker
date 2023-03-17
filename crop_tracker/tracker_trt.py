
import os
import sys
import random
import glob
import time
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple
from dataclasses import dataclass

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import tensorrt as trt

from supervision.color import ColorPalette
from supervision.dataclasses import Point
from supervision.detections import Detections, BoxAnnotator
from supervision.line_counter import LineCounter, LineCounterAnnotator


from tracker.byte_tracker import BYTETracker, STrack
from tracker.tracker_utils import box_iou_batch


# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
CLASS_NAMES = ['crop']

LINE_START = Point(10,1500)
LINE_END = Point(1000,1500)

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.8
    track_buffer: int = 30
    match_thresh: float = 0.9
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


class YV7TrackerTRT():
    def __init__(self, trt_model_path: str, device: str = 'cuda:0'):
        
        # Load all custom tensorrt plugins
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')

        self.device = torch.device(device)

        # Infer TensorRT Engine
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        with open(trt_model_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.bindings = OrderedDict()
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = model.create_execution_context()
        
        # warmup for 10 times
        for _ in range(10):
            tmp = torch.randn(1,3,640,640).to(self.device)
            self.binding_addrs['images'] = int(tmp.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))

    def infer_image(self, image_path: str, save_path:str, show:bool = False)->None:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        print(f'Loaded image shape: {im.shape}')
        im = torch.from_numpy(im).to(self.device)
        im/=255

        start = time.perf_counter()
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        print(f'Cost {time.perf_counter()-start} s')

        nums = self.bindings['num_dets'].data
        boxes = self.bindings['det_boxes'].data
        scores = self.bindings['det_scores'].data
        classes = self.bindings['det_classes'].data

        boxes = boxes[0,:nums[0][0]]
        scores = scores[0,:nums[0][0]]
        classes = classes[0,:nums[0][0]]

        colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(CLASS_NAMES)}
        for box,score,cl in zip(boxes,scores,classes):
            box = postprocess(box,ratio,dwdh).round().int()
            name = CLASS_NAMES[cl]
            color = colors[name]
            name += ' ' + str(round(float(score),3))
            cv2.rectangle(img,box[:2].tolist(),box[2:].tolist(),color,2)
            cv2.putText(img,name,(int(box[0]), int(box[1]) - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,color,thickness=2)

        Image.fromarray(img)

        if show:
            cv2.imshow('image',img)
            cv2.waitKey(0)

        if save_path:
            cv2.imwrite(save_path,img)

    def infer_video(self, video_path: str, save_path:str, show:bool = False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f'Video {video_path} not found or not openable')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(save_path, fourcc, fps, size)

        colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(CLASS_NAMES)}
        # for each frame in the video detect
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = frame.copy()
            image, ratio, dwdh = letterbox(image, auto=False)
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)
            im = image.astype(np.float32)
            im = torch.from_numpy(im).to(self.device)
            im/=255

            start = time.perf_counter()
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            print(f'Cost {time.perf_counter()-start} s')

            nums = self.bindings['num_dets'].data
            boxes = self.bindings['det_boxes'].data
            scores = self.bindings['det_scores'].data
            classes = self.bindings['det_classes'].data

            boxes = boxes[0,:nums[0][0]]
            scores = scores[0,:nums[0][0]]
            classes = classes[0,:nums[0][0]]

            
            for box,score,cl in zip(boxes,scores,classes):
                box = postprocess(box,ratio,dwdh).round().int()
                name = CLASS_NAMES[cl]
                color = colors[name]
                name += ' ' + str(round(float(score),3))
                cv2.rectangle(frame,box[:2].tolist(),box[2:].tolist(),color,2)
                cv2.putText(frame,name,(int(box[0]), int(box[1]) - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,color,thickness=2)
                

            if show:
                cv2.imshow('image',frame)
                cv2.waitKey(1)

            if save_path:
                video_writer.write(frame)

    
    def track_video(self, video_path, save_path, show=False)->None:
        
        # initialize dataset
        dataset = Video2Images(video_path)

        # initialize trackers
        byte_tracker = BYTETracker(BYTETrackerArgs())

        # initialize annotators
        box_annotator = BoxAnnotator(
            color=ColorPalette(),
            thickness=2,
            text_thickness=2,
            text_scale=2
            )
        line_counter = LineCounter(start=LINE_START, end=LINE_END)
        line_annotator = LineCounterAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=2
            )

        # initialize video writer
        if save_path:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid_writer = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (w, h)
                )

        # for each frame in the video detect
        for path, im0, vid_cap in dataset:
            image = im0.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image, ratio, dwdh = letterbox(image, auto=False)
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)
            im = image.astype(np.float32)
            im = torch.from_numpy(im).to(self.device)
            im/=255

            start = time.perf_counter()
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            

            nums = self.bindings['num_dets'].data
            boxes = self.bindings['det_boxes'].data
            scores = self.bindings['det_scores'].data
            class_ids = self.bindings['det_classes'].data

            boxes = boxes[0,:nums[0][0]]
            boxes = [postprocess(box,ratio,dwdh).round().int().cpu().numpy() for box in boxes]
            boxes = np.array(boxes)
            scores = scores[0,:nums[0][0]]
            class_ids = class_ids[0,:nums[0][0]]

            try:
                detections = Detections(
                        xyxy=boxes,
                        confidence=scores.cpu().numpy(),
                        class_id=class_ids.cpu().numpy(),
                    )
                tracks = byte_tracker.update(
                    output_results=detections2boxes(detections=detections),
                    img_info=im0.shape,
                    img_size=im0.shape
                )

                tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
                detections.tracker_id = np.array(tracker_id)
                
                # filtering out detections without trackers
                mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)
                # format custom labels

                labels = [
                    f"#{tracker_id} {CLASS_NAMES[class_id]} {confidence:0.2f}"
                    for _, confidence, class_id, tracker_id
                    in detections
                ]

                # updating line counter
                line_counter.update(detections=detections)
                # annotate and display frame
                im0 = box_annotator.annotate(frame=im0, detections=detections, labels=labels)
                line_annotator.annotate(frame=im0, line_counter=line_counter)
            
            except Exception as e:
                print(e)

            print(f'Inference-time(NMS): {round(time.perf_counter()-start,2)} s |Net crop count: {line_counter.get_count()}')

            if show:
                cv2.imshow('image',im0)
                cv2.waitKey(1)

            if save_path:
                vid_writer.write(im0)
        
        # release video writer and video capture
        if save_path:
            cap.release()
            vid_writer.release()
                
# util files for the tracker
#---------------------------
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[0:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def postprocess(boxes,r,dwdh):
    dwdh = torch.tensor(dwdh*2).to(boxes.device)
    boxes -= dwdh
    boxes /= r
    return boxes

def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))

# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)

# matches our bounding boxes with predictions
def match_detections_with_tracks(
        detections: Detections,
        tracks: List[STrack]
    ) -> Detections:
        if not np.any(detections.xyxy) or len(tracks) == 0:
            return np.empty((0,))

        tracks_boxes = tracks2boxes(tracks=tracks)
        iou = box_iou_batch(tracks_boxes, detections.xyxy)
        track2detection = np.argmax(iou, axis=1)
        
        tracker_ids = [None] * len(detections)
        
        for tracker_index, detection_index in enumerate(track2detection):
            if iou[tracker_index, detection_index] != 0:
                tracker_ids[detection_index] = tracks[tracker_index].track_id

        return tracker_ids


class Video2Images:
    def __init__(self, path: str):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')
        
        img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 
                       'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
        vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg',
                        'm4v', 'wmv', 'mkv']  # acceptable video suffixes
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'
        

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            #print(f'image {self.count}/{self.nf} {path}: ', end='')

        return path, img0, self.cap


if __name__ == '__main__':
    tracker = YV7TrackerTRT('/home/dadi_vardhan/Downloads/escarda/crop_trt/crop_yv7_960m.trt')
    # tracker.infer_image('/home/dadi_vardhan/Downloads/escarda/demo/test_crop.png',
    #                      save_path='/home/dadi_vardhan/Downloads/escarda/demo/test_out.jpg',
    #                     show=True)


    tracker.track_video('/home/dadi_vardhan/Downloads/escarda/camera_0.mp4',
                            save_path='/home/dadi_vardhan/Downloads/escarda/camera_0_out-trt.mp4',
                            show=False)