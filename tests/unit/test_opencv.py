import cv2
import numpy as np


def test_opencv_major_version_is_5() -> None:
    assert cv2.__version__.split(".")[0] == "5", cv2.__version__


def test_opencv_dnn_available() -> None:
    assert hasattr(cv2, "dnn")
    assert hasattr(cv2.dnn, "readNet")


def test_basic_image_analysis() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.rectangle(img, (16, 16), (48, 48), (255, 255, 255), -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert cv2.Laplacian(gray, cv2.CV_64F).var() > 0
