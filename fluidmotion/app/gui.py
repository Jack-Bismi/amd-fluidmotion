"""Qt6 GUI for FluidMotion preview and control."""

from typing import Optional
import numpy as np

try:
    from PyQt6 import QtWidgets, QtCore, QtGui
    HAS_QT = True
except ImportError:
    HAS_QT = False


class FluidGUI:
    """Minimal GUI for FluidMotion.

    Shows original vs processed side-by-side with FPS counter.
    Falls back to headless mode if Qt not available.
    """

    def __init__(self):
        self.has_gui = HAS_QT
        self._app = None
        self._window = None
        self._orig_label = None
        self._proc_label = None
        self._fps_label = None

    def start(self) -> None:
        if not HAS_QT:
            print("[GUI] Qt6 not installed. Running headless.")
            return

        self._app = QtWidgets.QApplication([])
        self._window = QtWidgets.QWidget()
        self._window.setWindowTitle("AMD-FluidMotion")
        self._window.resize(1280, 480)

        layout = QtWidgets.QHBoxLayout()

        orig_layout = QtWidgets.QVBoxLayout()
        orig_layout.addWidget(QtWidgets.QLabel("Original"))
        self._orig_label = QtWidgets.QLabel()
        self._orig_label.setMinimumSize(600, 400)
        self._orig_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        orig_layout.addWidget(self._orig_label)
        layout.addLayout(orig_layout)

        proc_layout = QtWidgets.QVBoxLayout()
        self._fps_label = QtWidgets.QLabel("FPS: --")
        proc_layout.addWidget(self._fps_label)
        self._proc_label = QtWidgets.QLabel()
        self._proc_label.setMinimumSize(600, 400)
        self._proc_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        proc_layout.addWidget(self._proc_label)
        layout.addLayout(proc_layout)

        self._window.setLayout(layout)
        self._window.show()

    def update(self, original: np.ndarray, processed: np.ndarray, fps: float) -> bool:
        """Update GUI with new frames. Returns False if closed."""
        if not self.has_gui:
            return True

        h, w, _ = original.shape
        qimg_orig = QtGui.QImage(original.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
        self._orig_label.setPixmap(QtGui.QPixmap.fromImage(qimg_orig).scaled(
            600, 400, QtCore.Qt.AspectRatioMode.KeepAspectRatio
        ))

        h, w, _ = processed.shape
        qimg_proc = QtGui.QImage(processed.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
        self._proc_label.setPixmap(QtGui.QPixmap.fromImage(qimg_proc).scaled(
            600, 400, QtCore.Qt.AspectRatioMode.KeepAspectRatio
        ))

        self._fps_label.setText(f"FPS: {fps:.1f}")
        QtCore.QCoreApplication.processEvents()

        return True

    def stop(self) -> None:
        if self.has_gui and self._window:
            self._window.close()
