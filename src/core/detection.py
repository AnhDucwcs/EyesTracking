import mediapipe as mp
import numpy as np


mp_face_mesh = mp.solutions.face_mesh

# Dò nét mặt và trích xuất lưới điểm đặc trưng
class FaceMeshDetector:
    def __init__(self, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.cfg = dict(max_num_faces=max_num_faces, refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence)
        self._mesh = mp_face_mesh.FaceMesh(**self.cfg)


    def process(self, frame_rgb):
        return self._mesh.process(frame_rgb)


    @staticmethod
    def landmarks_to_array(landmarks, image_width, image_height):
        pts = np.array([[lm.x * image_width, lm.y * image_height, lm.z] for lm in landmarks], dtype=np.float32)
        return pts