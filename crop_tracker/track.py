

import os
import sys
import argparse
from pathlib import Path
import random
import time
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass

# configuring paths due to poor packaging of byte-tracker
HOME = os.getcwd()
sys.path.append(f"{HOME}/yolov7")


import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator


from tracker.byte_tracker import BYTETracker, STrack
from tracker_utils import box_iou_batch

# change directory to yolov7
os.chdir(f"{HOME}/yolov7")

from utils.general import increment_path, set_logging, check_img_size
from yolov7.utils.general import check_imshow, non_max_suppression
from yolov7.utils.general import scale_coords
from yolov7.utils.torch_utils import select_device, TracedModel, time_synchronized
from yolov7.models.experimental import attempt_load
#from yolov7.utils.plots import plot_one_box
from yolov7.utils.datasets import LoadStreams, LoadImages

# LINE_START: Point = Point(50, 50)
# LINE_END: Point = Point(50, 200)
LINE_START = Point(10,1500)
LINE_END = Point(1000,1500)

# LINE_START: Point = Point(500, 500)
# LINE_END: Point = Point(500, 1500)
# CLASS_ID = [2, 3, 5, 7] # cars, buses, trucks, motorbikes
#CLASS_ID = [41] # person:0, cups:41
CLASS_ID = [0] # crop:0

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.8
    track_buffer: int = 30
    match_thresh: float = 0.9
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# converts Detections into format that can be consumed by match_detections_with_tracks function
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

class YoloV7Tracker:
    def __init__(self, opt, save_img = False):
        self.opt = opt
        self.save_img = not opt.nosave and not opt.source.endswith('.txt')
        self.source = opt.source
        self.weights = opt.weights
        self.view_img = opt.view_img
        self.save_txt = opt.save_txt
        self.imgsz = opt.img_size
        self.trace = not opt.no_trace
        self.save_dir = self.get_save_dir()
        set_logging()
        self.device = select_device(opt.device)
        self.model, self.stride, self.half = self.load_yolo_model()
        self.classes = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.classes]
        self.vid_path, self.vid_writer = None, None
        self.tracker_dataset, self.webcam, self.view_img  = self.tracker_dataset_generator()
        self.byte_tracker = BYTETracker(BYTETrackerArgs())
        self.box_annotator = BoxAnnotator(
            color=ColorPalette(),
            thickness=4,
            text_thickness=3,
            text_scale=2
            )
        self.line_counter = LineCounter(start=LINE_START, end=LINE_END)
        self.line_annotator = LineCounterAnnotator(
            thickness=4,
            text_thickness=3,
            text_scale=2
            )
        

    def get_save_dir(self)->str:
        save_dir = Path(increment_path(Path(self.opt.project) / self.opt.name,
                         exist_ok=self.opt.exist_ok))  # increment run
        (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True,
                                                     exist_ok=True)  # make dir
        return save_dir

    def load_yolo_model(self):
        model = attempt_load(self.weights, map_location=self.device) # load FP32 model
        stride = int(model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check img_size
        half = self.device.type != 'cpu'  # half precision only supported on CUDA
        
        if self.trace:
            model = TracedModel(model, self.device, self.imgsz)

        if half:
            model.half()  # to FP16
        
        return model, stride, half
    
    def tracker_dataset_generator(self)-> Tuple[LoadImages, bool, bool]:
        webcam = self.source.isnumeric() \
        or self.source.endswith('.txt')  \
        or self.source.lower().startswith(('rtsp://', 'rtmp://','http://', 'https://'))

        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(
                self.source,
                img_size=self.imgsz,
                stride=self.stride
                )

        else:
            #generator =  get_video_frames_generator(self.source)
            dataset = LoadImages(
                self.source,
                img_size=self.imgsz,
                stride=self.stride
                )
        
        try:
            return dataset, webcam, view_img
        except UnboundLocalError:
            return dataset, webcam, False
        

    def run_tracker_inference(self):
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
            old_img_w = old_img_h = self.imgsz
            old_img_b = 1
        for path, img, im0s, vid_cap in self.tracker_dataset:
            # convert frame to tensor
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # model prediction on single frame and conversion to supervision Detections
            # Inference
            t1 = time_synchronized()
            with torch.no_grad():
                pred = self.model(img,augment=opt.augment)[0]
            t2 = time_synchronized()
            # Apply NMS
            pred = non_max_suppression(
                pred, opt.conf_thres, opt.iou_thres,
                classes=opt.classes, agnostic=opt.agnostic_nms
                )
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), self.tracker_dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(self.tracker_dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(self.save_dir / p.name)  # img.jpg
            # txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.tracker_dataset.mode == 'image' else f'_{frame}')  # img.txt
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {self.classes[int(c)]}{'s' * (n > 1)}, "  # add to string

            #write results
            boxes =[]
            confs=[]
            class_ids=[]
            for *xyxy, conf, cls in reversed(det):
                boxes.append(torch.tensor(xyxy).detach().cpu().numpy())
                confs.append(torch.tensor(conf).detach().cpu().numpy())
                class_ids.append(torch.tensor(cls).detach().cpu().numpy().astype(int))
            try:
                detections = Detections(
                        xyxy=np.array(boxes),
                        confidence=np.array(confs),
                        class_id=np.array(class_ids),
                    )
                    # filtering out detections with unwanted classes
                # mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
                # detections.filter(mask=mask, inplace=True)
                # tracking detections
                tracks = self.byte_tracker.update(
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
                    f"#{tracker_id} {self.classes[class_id]} {confidence:0.2f}"
                    for _, confidence, class_id, tracker_id
                    in detections
                ]
                # updating line counter
                self.line_counter.update(detections=detections)
                # annotate and display frame
                im0 = self.box_annotator.annotate(frame=im0, detections=detections, labels=labels)
                self.line_annotator.annotate(frame=im0, line_counter=self.line_counter)
            
            except Exception as e:
                print(e)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if self.view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with tracked objects)
            if self.save_img:
                if self.tracker_dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if self.vid_path != save_path:  # new video
                        self.vid_path = save_path
                        if isinstance(self.vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps,
                            (w, h)
                            )
                    vid_writer.write(im0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    yolo_obj = YoloV7Tracker(opt)
    
    with torch.no_grad():
        yolo_obj.run_tracker_inference()