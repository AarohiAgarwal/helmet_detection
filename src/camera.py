"""
Threaded Camera and Video Stream Reader for Real-Time Helmet Detection.
Provides high-performance, non-blocking frame retrieval from webcams and video files.
"""

import time
import threading
from pathlib import Path
from typing import Union, Tuple, Optional
import cv2
import numpy as np


class CameraStream:
    """
    Manages non-blocking video frame capture from a webcam index or video file.
    Runs frame capture in a dedicated background thread to prevent IO bottlenecks.
    """

    def __init__(self, source: Union[int, str] = 0, target_fps: Optional[float] = None):
        self.source = int(source) if isinstance(source, str) and source.isdigit() else source
        self.target_fps = target_fps
        self.cap = None
        self.frame = None
        self.ret = False
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.is_file = isinstance(self.source, str) and not self.source.isdigit()

        self._init_capture()

    def _init_capture(self):
        """Initializes OpenCV VideoCapture with hardware-appropriate backend."""
        try:
            if isinstance(self.source, int):
                # On Windows, cv2.CAP_DSHOW provides fast camera startup
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    # Fallback to default backend
                    self.cap = cv2.VideoCapture(self.source)
            else:
                src_path = Path(self.source)
                if not src_path.exists():
                    raise FileNotFoundError(f"Video file not found: {self.source}")
                self.cap = cv2.VideoCapture(str(src_path))

            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Failed to open video source '{self.source}'. "
                    f"Ensure webcam is connected, permissions are granted, or video path is valid."
                )

            # Query stream parameters
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

            # Read initial probe frame
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                print(f"[Warning] Successfully opened source {self.source}, but could not read probe frame.")

        except Exception as e:
            self.release()
            raise e

    def start(self) -> "CameraStream":
        """Starts background reader thread."""
        if self.running:
            return self

        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        return self

    def _update_loop(self):
        """Continuously grabs latest frames in a background thread."""
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if not ret:
                # End of stream (e.g. video file completed)
                with self.lock:
                    self.ret = False
                if self.is_file:
                    self.running = False
                time.sleep(0.01)
                break

            with self.lock:
                self.ret = ret
                self.frame = frame

            # Yield CPU slightly for file playback synchronization
            if self.is_file and self.fps > 0:
                time.sleep(1.0 / (self.fps * 1.2))
            else:
                time.sleep(0.001)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the most recent frame."""
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        """Signals background thread to stop."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def release(self):
        """Releases all camera and thread resources."""
        self.stop()
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

