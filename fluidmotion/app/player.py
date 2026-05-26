"""Video player with real-time FluidMotion processing."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Generator


class VideoPlayer:
    """Video file reader with frame buffering.

    Handles decoding and frame pair generation.
    """

    def __init__(self, video_path: Path, prefetch: int = 32):
        self.video_path = Path(video_path)
        self.prefetch = prefetch

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._total_frames: int = 0
        self._width: int = 0
        self._height: int = 0

    def open(self) -> None:
        self._cap = cv2.VideoCapture(str(self.video_path))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Yield frames one by one."""
        if not self._cap or not self._cap.isOpened():
            self.open()

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> tuple:
        return (self._width, self._height)

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def close(self) -> None:
        if self._cap:
            self._cap.release()
