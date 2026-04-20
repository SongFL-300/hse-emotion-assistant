import cv2
import numpy as np
from pathlib import Path
import os
import sys
import math

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("MediaPipe is not installed. Please run: pip install mediapipe")

class FaceMeshDetector:
    """
    基于 MediaPipe Face Mesh 的高精度人脸关键点检测与对齐模块
    适用于微表情识别、表情分析、人脸 ROI 提取等任务
    """

    # 预定义关键点索引（MediaPipe 官方）
    FACEMESH_LIPS = frozenset([
        (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
        (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
        (61, 185), (185, 40), (40, 39), (39, 37), (37, 0),
        (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
        (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
        (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
        (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
        (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
    ])

    FACEMESH_LEFT_EYE = frozenset([
        (263, 249), (249, 390), (390, 373), (373, 374), (374, 380),
        (380, 381), (381, 382), (382, 362), (263, 466), (466, 388),
        (388, 387), (387, 386), (386, 385), (385, 384), (384, 398),
        (398, 362)
    ])

    FACEMESH_RIGHT_EYE = frozenset([
        (33, 7), (7, 163), (163, 144), (144, 145), (145, 153),
        (153, 154), (154, 155), (155, 133), (33, 246), (246, 161),
        (161, 160), (160, 159), (159, 158), (158, 157), (157, 173),
        (173, 133)
    ])

    # 用于仿射对齐的5个关键点（左眼、右眼、鼻尖、左嘴角、右嘴角）
    ALIGNMENT_POINTS = {
        'left_eye_center': [33, 133, 157, 158, 159, 160, 161, 246],  # 右眼（图像左侧）
        'right_eye_center': [263, 362, 385, 386, 387, 388, 466, 398],  # 左眼（图像右侧）
        'nose_tip': [1],
        'mouth_left': [61, 146, 91, 181, 84],
        'mouth_right': [314, 405, 321, 375, 291]
    }

    def __init__(
            self,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_size=(112, 112),
            offline_mode=True,
    ):
        # ✅ 在最开始就初始化属性，防止 __del__ 出错
        self._is_closed = False
        self.face_mesh = None

        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.output_size = output_size
        self.offline_mode = offline_mode

        # ✅ 修复：查找所有可能的模型文件
        if offline_mode:
            model_path = self._find_model_path()
            if not model_path.exists():
                raise FileNotFoundError(
                    f"❌ 离线模式下未找到 Face Mesh 模型:\n{model_path}\n"
                    f"可能的路径：\n"
                    f"  - C:\\Users\\<用户名>\\AppData\\Local\\MediaPipe\\modules\\face_landmark\\face_landmark.tflite\n"
                    f"  - C:\\Users\\<用户名>\\AppData\\Local\\MediaPipe\\modules\\face_landmark\\face_landmark_with_attention.tflite\n"
                    "请先运行 offline_mode=False 一次，或手动放置模型文件。"
                )

        # 初始化 MediaPipe FaceMesh
        if not hasattr(mp, 'solutions'):
            raise ImportError(
                "Your MediaPipe installation is corrupted (missing 'solutions' module).\n"
                "Current mediapipe path: {}\n"
                "Please reinstall it manually.".format(mp.__path__)
            )
            
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _get_cache_dir(self):
        """获取 MediaPipe 缓存目录"""
        if os.name == "nt":  # Windows
            return Path(os.environ.get("LOCALAPPDATA", "")) / "MediaPipe"
        else:  # Linux / macOS
            return Path.home() / ".cache" / "mediapipe"

    def _find_model_path(self):
        """查找 Face Mesh 模型文件（尝试多个可能的路径）"""
        cache_dir = self._get_cache_dir()

        # 可能的模型文件名（MediaPipe 可能使用不同名称）
        possible_names = [
            "face_landmark_with_attention.tflite",
            "face_landmark.tflite",
            "face_landmark_front.tflite"
        ]

        for name in possible_names:
            path = cache_dir / "modules" / "face_landmark" / name
            if path.exists():
                return path

        # 如果在 face_landmark 子目录没找到，尝试根目录
        for name in possible_names:
            path = cache_dir / "modules" / name
            if path.exists():
                return path

        # 如果都没找到，返回最常见的路径
        return cache_dir / "modules" / "face_landmark" / "face_landmark_with_attention.tflite"

    def detect(self, frame):
        """
        仅检测关键点，不进行对齐

        Args:
            frame (np.ndarray): BGR 图像 (H, W, 3)

        Returns:
            List[Dict]: 每个人脸包含：
                - 'landmarks': 归一化坐标列表 [(x, y, z), ...] 长度 468 或 478
                - 'landmarks_px': 像素坐标 [(x, y), ...]
        """
        if self._is_closed or self.face_mesh is None or frame is None or frame.size == 0:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        results = self.face_mesh.process(rgb)

        faces = []
        if results.multi_face_landmarks:
            for lmks in results.multi_face_landmarks:
                norm_lmk_list = [(p.x, p.y, p.z) for p in lmks.landmark]
                px_lmk_list = [(int(p.x * w), int(p.y * h)) for p in lmks.landmark]
                faces.append({
                    'landmarks': norm_lmk_list,
                    'landmarks_px': px_lmk_list,
                })
        return faces

    def align_face(self, frame, face_index=0):
        """
        对指定人脸进行仿射对齐，输出标准化图像

        Args:
            frame (np.ndarray): BGR 输入帧
            face_index (int): 使用第几个人脸（默认0）

        Returns:
            aligned_rgb (np.ndarray or None): 对齐后的 RGB 图像 (H, W, 3)
            landmarks (list or None): 像素坐标关键点
        """
        faces = self.detect(frame)
        if not faces or face_index >= len(faces):
            return None, None

        lmks_px = faces[face_index]['landmarks_px']
        h, w = frame.shape[:2]

        try:
            left_eye = self._get_center(lmks_px, self.ALIGNMENT_POINTS['left_eye_center'])
            right_eye = self._get_center(lmks_px, self.ALIGNMENT_POINTS['right_eye_center'])
            nose_tip = self._get_center(lmks_px, self.ALIGNMENT_POINTS['nose_tip'])
            mouth_left = self._get_center(lmks_px, self.ALIGNMENT_POINTS['mouth_left'])
            mouth_right = self._get_center(lmks_px, self.ALIGNMENT_POINTS['mouth_right'])
        except IndexError:
            return None, None  # 关键点缺失

        aligned_bgr = self._warp_face_with_quad(
            frame=frame,
            left_eye=left_eye,
            right_eye=right_eye,
            nose_tip=nose_tip,
            mouth_left=mouth_left,
            mouth_right=mouth_right,
        )
        if aligned_bgr is None:
            return None, None
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

        return aligned_rgb, lmks_px

    def estimate_pose(self, landmarks_px, frame_shape):
        if not landmarks_px:
            return {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "frontal_score": 0.0,
            }

        try:
            left_eye = self._get_center(landmarks_px, self.ALIGNMENT_POINTS['left_eye_center'])
            right_eye = self._get_center(landmarks_px, self.ALIGNMENT_POINTS['right_eye_center'])
            nose_tip = self._get_center(landmarks_px, self.ALIGNMENT_POINTS['nose_tip'])
            mouth_left = self._get_center(landmarks_px, self.ALIGNMENT_POINTS['mouth_left'])
            mouth_right = self._get_center(landmarks_px, self.ALIGNMENT_POINTS['mouth_right'])
        except Exception:
            return {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "frontal_score": 0.0,
            }

        if left_eye[0] > right_eye[0]:
            left_eye, right_eye = right_eye, left_eye

        eye_mid = (left_eye + right_eye) * 0.5
        mouth_mid = (mouth_left + mouth_right) * 0.5
        eye_dist = max(1.0, float(np.linalg.norm(right_eye - left_eye)))
        mouth_width = max(1.0, float(np.linalg.norm(mouth_right - mouth_left)))
        vertical_span = max(1.0, float(np.linalg.norm(mouth_mid - eye_mid)))

        roll = math.degrees(math.atan2(float(right_eye[1] - left_eye[1]), float(right_eye[0] - left_eye[0])))
        yaw = float((nose_tip[0] - eye_mid[0]) / max(1.0, eye_dist * 0.5))
        pitch_ratio = float((nose_tip[1] - eye_mid[1]) / vertical_span)
        pitch = (pitch_ratio - 0.58) / 0.35

        symmetry = mouth_width / eye_dist
        symmetry_penalty = min(1.0, abs(symmetry - 0.95) / 0.45)
        yaw_penalty = min(1.0, abs(yaw) / 0.90)
        pitch_penalty = min(1.0, abs(pitch) / 0.95)
        roll_penalty = min(1.0, abs(roll) / 28.0)

        frontal_score = 1.0 - (yaw_penalty * 0.40 + pitch_penalty * 0.35 + roll_penalty * 0.15 + symmetry_penalty * 0.10)
        frontal_score = max(0.0, min(1.0, float(frontal_score)))

        return {
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll / 30.0),
            "frontal_score": frontal_score,
        }

    def _get_center(self, landmarks_px, indices):
        pts = np.array([landmarks_px[i] for i in indices], dtype=np.float32)
        return np.mean(pts, axis=0)

    def _warp_face_with_quad(self, frame, *, left_eye, right_eye, nose_tip, mouth_left, mouth_right):
        out_w, out_h = self.output_size

        if left_eye[0] > right_eye[0]:
            left_eye, right_eye = right_eye, left_eye
            mouth_left, mouth_right = mouth_right, mouth_left

        eye_avg = (left_eye + right_eye) * 0.5
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_eye = right_eye - left_eye
        eye_to_mouth = mouth_avg - eye_avg

        # FFHQ-style oriented crop: adaptively follows the real face pose instead of forcing a fixed 3-point template.
        x_axis = eye_to_eye - np.array([-eye_to_mouth[1], eye_to_mouth[0]], dtype=np.float32)
        norm = float(np.linalg.norm(x_axis))
        if norm < 1e-6:
            return None
        x_axis /= norm
        scale = max(float(np.linalg.norm(eye_to_eye)) * 2.2, float(np.linalg.norm(eye_to_mouth)) * 2.0)
        x_axis *= scale
        y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)
        center = eye_avg + eye_to_mouth * 0.12

        quad = np.stack(
            [
                center - x_axis - y_axis,
                center - x_axis + y_axis,
                center + x_axis + y_axis,
                center + x_axis - y_axis,
            ]
        ).astype(np.float32)

        dst = np.array(
            [
                [0, 0],
                [0, out_h - 1],
                [out_w - 1, out_h - 1],
                [out_w - 1, 0],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(quad, dst)
        return cv2.warpPerspective(
            frame,
            matrix,
            (out_w, out_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def draw_landmarks(self, image, landmarks_px, color=(0, 255, 0), thickness=1):
        """在图像上绘制关键点和连接线（用于调试）"""
        if not landmarks_px:
            return image

        # 绘制点
        for x, y in landmarks_px:
            cv2.circle(image, (x, y), 1, color, -1)

        # 绘制连接线（简化版）
        if hasattr(self.mp_face_mesh, 'FACEMESH_TESSELATION'):
            connections = list(self.mp_face_mesh.FACEMESH_TESSELATION)
            for start_idx, end_idx in connections:
                start = landmarks_px[start_idx]
                end = landmarks_px[end_idx]
                cv2.line(image, start, end, color, thickness)

        return image

    def release(self):
        """释放资源"""
        if not self._is_closed and self.face_mesh is not None:
            self.face_mesh.close()
            self._is_closed = True

    def __del__(self):
        self.release()
