#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np
import torch

from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView

def main():
    source = "sample_video.mp4"
    output_path = "output.mp4"
    depth_output_path = "depth_output.mp4"

    yolo_model_size = "nano"
    depth_model_size = "small"
    device = 'cpu'

    conf_threshold = 0.25
    iou_threshold = 0.45
    classes = None

    enable_tracking = True
    enable_bev = True

    print(f"Using device: {device}")
    print("Initializing models...")

    detector = ObjectDetector(yolo_model_size, conf_threshold, iou_threshold, classes, device)
    depth_estimator = DepthEstimator(depth_model_size, device)
    bbox3d_estimator = BBox3DEstimator()
    bev = BirdEyeView(scale=60, size=(300, 300)) if enable_bev else None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    depth_writer = cv2.VideoWriter(depth_output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    print("Starting processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = frame.copy()

        detection_frame, detections = detector.detect(frame, track=enable_tracking)

        # Depth Estimation
        depth_map = depth_estimator.estimate_depth(frame)
        depth_colored = depth_estimator.colorize_depth(depth_map)

        boxes_3d = []
        active_ids = []

        for det in detections:
            bbox, score, class_id, obj_id = det
            class_name = detector.get_class_names()[class_id]

            # Get depth value (median in bbox)
            depth_value = depth_estimator.get_depth_in_region(depth_map, bbox)

            box_3d = {
                'bbox_2d': bbox,
                'depth_value': depth_value,
                'class_name': class_name,
                'object_id': obj_id,
                'score': score
            }
            boxes_3d.append(box_3d)

            if obj_id is not None:
                active_ids.append(obj_id)

        bbox3d_estimator.cleanup_trackers(active_ids)

        for box in boxes_3d:
            result_frame = bbox3d_estimator.draw_box_3d(result_frame, box)

        if enable_bev:
            bev.reset()
            for box in boxes_3d:
                bev.draw_box(box)
            bev_image = bev.get_image()

            bev_height = height // 3
            bev_width = bev_height

            if bev_height > 0 and bev_width > 0:
                bev_resized = cv2.resize(bev_image, (bev_width, bev_height))
                result_frame[height - bev_height:height, 0:bev_width] = bev_resized
                cv2.rectangle(result_frame, (0, height - bev_height), (bev_width, height), (255, 255, 255), 1)
                cv2.putText(result_frame, "Bird's Eye View", (10, height - bev_height + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps_value = frame_count / elapsed
        else:
            fps_value = "--"

        cv2.putText(result_frame, f"FPS: {fps_value}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out.write(result_frame)
        depth_writer.write(cv2.resize(depth_colored, (width, height)))

        cv2.imshow("3D Object Detection + BEV", result_frame)
        cv2.imshow("Depth Map", depth_colored)

        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    out.release()
    depth_writer.release()
    cv2.destroyAllWindows()
    print("Processing complete. Outputs saved.")

if __name__ == "__main__":
    main()
