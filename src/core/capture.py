import threading
import cv2
import time


# Xử lý việc lấy khung hình từ camera trong một luồng riêng biệt
class Camera:
    def __init__(self, device=0, width=1440, height=900, fps=30):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None
        self._thread = None
        self._running = False
        self._frame = None
        self._lock = threading.Lock()


    def start(self):
        if self._running:
            return
        self._cap = cv2.VideoCapture(self.device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()


    def _update(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame


    def read(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()



    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()