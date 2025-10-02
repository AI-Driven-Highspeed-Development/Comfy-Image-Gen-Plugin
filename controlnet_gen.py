import numpy as np
import cv2
from typing import Tuple
from plugins.yolo_pose_plugin.data_structure import (
    PoseData,
    Skeleton,
    Joint,
    COCO_CONNECTIONS,
    COCO_CONNECTION_COLORS,
    COCO_KEYPOINT_NAMES,
)
from managers.config_manager.config_manager import ConfigManager


class ControlNetGen:
    _instance = None  # singleton storage

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):  # prevent re-init on subsequent instantiations
            return
        self._initialized = True
        """Generator for ControlNet OpenPose-style conditioning images (singleton)."""
        self.config = ConfigManager().config
        self.img_cfg = self.config.comfy_image_gen_plugin.controlnet.image
        self.yolo_cfg = self.img_cfg.yolo
        
    

    def yolo_control_net_gen(self, pose_data: PoseData) -> np.ndarray:
        """Render a ControlNet-style pose conditioning image from YOLO Pose data.

        The input PoseData uses normalized joint coordinates (0..1). This will
        draw COCO skeleton connections with per-connection colors on a blank image.

        Returns:
            np.ndarray: BGR image of shape (H, W, 3), dtype=uint8.
        """
        # Start with a solid background
        img = np.full((self.img_cfg.height, self.img_cfg.width, 3), self.yolo_cfg.background_color, dtype=np.uint8)

        if pose_data is None or not pose_data.skeletons:
            return img

        def to_px(x_norm: float, y_norm: float) -> Tuple[int, int]:
            x = int(np.clip(x_norm, 0.0, 1.0) * (self.img_cfg.width - 1))
            y = int(np.clip(y_norm, 0.0, 1.0) * (self.img_cfg.height - 1))
            return x, y

        for sk in pose_data.skeletons:
            # Build an array of joint (x,y,conf) in COCO index order when possible
            # Fallback to matching by label if order is unknown
            idx_by_name = {name: i for i, name in enumerate(COCO_KEYPOINT_NAMES)}
            joints_xyc = [(None, None, 0.0)] * len(COCO_KEYPOINT_NAMES)

            for j in sk.joints:
                if j.label in idx_by_name:
                    i = idx_by_name[j.label]
                    joints_xyc[i] = (j.x, j.y, j.confidence)

            # Draw connections
            for conn_idx, (a, b) in enumerate(COCO_CONNECTIONS):
                if a >= len(joints_xyc) or b >= len(joints_xyc):
                    continue
                xa, ya, ca = joints_xyc[a]
                xb, yb, cb = joints_xyc[b]
                if xa is None or xb is None:
                    continue
                if ca < self.yolo_cfg.conf_threshold or cb < self.yolo_cfg.conf_threshold:
                    continue
                pa = to_px(xa, ya)
                pb = to_px(xb, yb)
                color = COCO_CONNECTION_COLORS[conn_idx] if conn_idx < len(COCO_CONNECTION_COLORS) else (255, 255, 255)
                cv2.line(img, pa, pb, color, self.yolo_cfg.line_thickness, lineType=cv2.LINE_AA)

            # Draw joints as small circles for visual clarity
            for i, (x, y, c) in enumerate(joints_xyc):
                if x is None or c < self.yolo_cfg.conf_threshold:
                    continue
                p = to_px(x, y)
                cv2.circle(img, p, self.yolo_cfg.joint_radius, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

        return img


    def _yolo_control_net_debug_gen(self) -> np.ndarray:
        """Build a plausible single-person PoseData for debugging rendering."""
        cx = 0.5
        nose_y = 0.18
        shoulder_y = 0.30
        elbow_y = 0.42
        wrist_y = 0.55
        hip_y = 0.45
        knee_y = 0.65
        ankle_y = 0.85

        dx_sh = 0.12
        dx_el = 0.18
        dx_wr = 0.22
        dx_hip = 0.09
        dx_knee = 0.08
        dx_ankle = 0.07

        pts = {
            "nose": (cx, nose_y),
            "left_eye": (cx - 0.02, nose_y + 0.01),
            "right_eye": (cx + 0.02, nose_y + 0.01),
            "left_ear": (cx - 0.06, nose_y + 0.02),
            "right_ear": (cx + 0.06, nose_y + 0.02),
            "left_shoulder": (cx - dx_sh, shoulder_y),
            "right_shoulder": (cx + dx_sh, shoulder_y),
            "left_elbow": (cx - dx_el, elbow_y),
            "right_elbow": (cx + dx_el, elbow_y),
            "left_wrist": (cx - dx_wr, wrist_y),
            "right_wrist": (cx + dx_wr, wrist_y),
            "left_hip": (cx - dx_hip, hip_y),
            "right_hip": (cx + dx_hip, hip_y),
            "left_knee": (cx - dx_knee, knee_y),
            "right_knee": (cx + dx_knee, knee_y),
            "left_ankle": (cx - dx_ankle, ankle_y),
            "right_ankle": (cx + dx_ankle, ankle_y),
        }

        joints = []
        xs, ys = [], []
        for name in COCO_KEYPOINT_NAMES:
            x, y = pts.get(name, (cx, shoulder_y))
            x = float(np.clip(x, 0.0, 1.0))
            y = float(np.clip(y, 0.0, 1.0))
            xs.append(x); ys.append(y)
            joints.append(Joint(x=x, y=y, confidence=1.0, label=name))

        xmin = max(0.0, min(xs) - 0.02)
        ymin = max(0.0, min(ys) - 0.02)
        xmax = min(1.0, max(xs) + 0.02)
        ymax = min(1.0, max(ys) + 0.02)

        sk = Skeleton(joints=joints, confidence=0.95, bounding_box=(xmin, ymin, xmax, ymax))
        debug_pose_data = PoseData(skeletons=[sk])
        return self.yolo_control_net_gen(debug_pose_data)