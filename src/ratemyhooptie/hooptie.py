import sys
import cv2
import PIL as pil
from PIL import ImageQt
from PIL import Image
import io
import base64
import textwrap
from openai import OpenAI
import subprocess as sp
import random
import numpy as np
import argparse
import pathlib
import abc

from PyQt6 import QtCore as qt

from PyQt6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QHBoxLayout,
        QVBoxLayout,
        QStackedLayout,
        QPushButton,
        QLabel,
		QGraphicsView,
        QGraphicsScene,
        QFrame,
        QSizePolicy,
        QFileDialog,
        QMessageBox,
)

from PyQt6.QtMultimedia import (
        QCamera,
        QMediaCaptureSession,
        QImageCapture
)

from PyQt6.QtMultimediaWidgets import (
        QVideoWidget,
        QGraphicsVideoItem,
)

from PyQt6.QtGui import QImage, QPixmap, QFont


usage = """
Usage:
    Press space to take a picture of your hooptie. The picture and the ARTIFICIAL INTELLIGENCEs rating will be displayed (probably). Press ENTER to replay the message or any other key to rate a new hooptie.

    Your hooptie and its rating will be saved for future enjoyment (use the --output argument to control where).
"""


class Critic(abc.ABC):
    @abc.abstractmethod
    def rate(self, img: QImage) -> str:
        pass


class ReplayCritic(Critic):
    def __init__(self, msg_path: pathlib.Path):
        with open(msg_path, 'r') as f:
            self.msg = f.read()

    def rate(self, img: QImage) -> str:
        return self.msg


class OpenaiCritic(Critic):
    def __init__(self):
        self.client = OpenAI()

    def rate(self, img: pil.Image) -> str:
        img = img.resize((512, 512)).convert("RGB")

        b = io.BytesIO()
        img.save(b, format="JPEG")
        img_str = base64.b64encode(b.getvalue()).decode("utf-8")

        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "You are an expert in automotive engineering. You have a sardonic personality and are prone to witty commentary, you have some favorite models which these cars are not. Your goal is to accurately identify the automobile in this picture and then say something provacative and amusing about the automobile. You should make fun of its owners and question their taste and judgment for bringing the car here. You should end with something amusing and cheeky. You are allowed to ocasionally say nice things. "},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}",
                                "detail": "low"
                            },
                        }
                    ],
                }
            ],
            max_tokens=150
        )

        # GPT4 sometimes ends mid-sentence. Just cut off the last incomplete sentence.
        msg = completion.choices[0].message.content
        msg = msg[:msg.rfind(".")+1]
        return(msg)


def apply_msg(img: np.ndarray, msg: str):
    textImg = img.copy()
    height, width, channel = textImg.shape

    font = cv2.FONT_HERSHEY_SIMPLEX

    # wrapped = textwrap.wrap(msg, width=38)
    wrapped = textwrap.wrap(msg, width=45)
    x, y = 10, 40
    # font_size = 1
    font_size = 1.4
    font_thickness = 2

    for i, line in enumerate(wrapped):
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]

        gap = textsize[1] + 10

        # y = int((textImg.shape[0] + textsize[1]) / 2) + i * gap
        y = 50 + i * gap
        x = int((textImg.shape[1] - textsize[0]) / 2)

        cv2.putText(textImg, line, (x, y), font,
                    font_size,
                    (255,255,255),
                    font_thickness,
                    lineType = cv2.LINE_AA)

    return textImg


def display(
        img: np.ndarray,
        msg: str,
        windowName:str,
        silent: bool =False
) -> np.ndarray:
    """Display the image and message, and read the message. Press "enter" to
    replay, or any other key to exit."""
    # textBox = np.zeros([1080,675,3],dtype=np.uint8)
    textBox = np.zeros([1920,1280,3],dtype=np.uint8)
    textBox.fill(0)
    textBox = apply_msg(textBox, msg)

    # imgBig = cv2.resize(img, (1080,1080))
    imgBig = cv2.resize(img, (1920,1920))
    textImg = np.concatenate((imgBig, textBox), axis=1)

    # 13 is enter
    k = 13
    while k == 13:
        cv2.imshow(windowName, textImg)
        if not silent:
            speechProc = sp.Popen(["say", "-v", "Daniel", "-r", "175", msg])

        k = cv2.waitKey(0)

        if not silent:
            speechProc.wait()

    return textImg


def crop_square(img: np.ndarray):
    cropped = img.copy()

    if cropped.shape[1] == cropped.shape[0]:
        pass
    elif cropped.shape[1] > cropped.shape[0]:
        # Crop assuming the raw frame is wider than it is tall
        xMargin = int((cropped.shape[1] - cropped.shape[0]) / 2)
        cropped = cropped[:, xMargin:-xMargin].copy()
    else:
        # Crop assuming the raw frame is taller than it is wide
        yMargin = int((cropped.shape[0] - cropped.shape[1]) / 2)
        cropped = cropped[yMargin:-yMargin, :]

    return cropped


# def main():
#     parser = argparse.ArgumentParser(description="Rate your hooptie with the power of ARTIFICIAL INTELLIGENCE!")
#     parser.add_argument("-f", "--file", type=pathlib.Path, help="Rate a static image rather than using the camera. If there is a FILENAME.txt in the same directory, that will be used as the rating rather than asking the ARTIFICIAL INTELLIGENCE!.")
#     parser.add_argument("-o", "--output", type=pathlib.Path, help="Save the joint img+prediction to the specified file.")
#     parser.add_argument("-s", "--silent", action="store_true", help="Supress the totally badass robot voice.")
#
#     args = parser.parse_args()
#
#     print(usage)
#
#     imgWindow = "Your Hooptie"
#     cv2.namedWindow(imgWindow, cv2.WINDOW_NORMAL)
#
#     if args.file is not None:
#         imgPath = args.file.resolve()
#         msgPath = imgPath.with_suffix(".txt")
#
#         img = cv2.imread(str(imgPath))
#         img = crop_square(img)
#
#         if msgPath.exists():
#             with open(msgPath, 'r') as f:
#                 msg = f.read()
#         else:
#             OAIClient = OpenAI()
#             msg = getPrediction(img, OAIClient)
#
#         msgImg = display(img, msg, imgWindow, args.silent)
#
#         if args.output is not None:
#             cv2.imwrite(str(args.output), msgImg)
#     else:
#         cam = cv2.VideoCapture(0)
#         OAIClient = OpenAI()
#         while True:
#             ret, frame = cam.read()
#             if not ret:
#                 raise RuntimeError("Camera failed!")
#
#             frame = crop_square(frame)
#             cv2.imshow(imgWindow, frame)
#             k = cv2.waitKey(33)
#             if k == 32:
#                 name = str(random.randint(0, 10e9))
#                 msg = getPrediction(frame, OAIClient)
#
#                 outDir = "thill24_race1/"
#                 with open(outDir + name + ".txt", 'w') as f:
#                     f.write(msg)
#                 cv2.imwrite(outDir + name + ".png", frame)
#
#                 msgImg = display(frame, msg, imgWindow, args.silent)
#                 if args.output is not None:
#                     cv2.imwrite(str(args.output), msgImg)
#             elif k == 27:
#                 break


# from https://stackoverflow.com/questions/69183307/how-to-display-image-in-ratio-as-preserveaspectfit-in-qt-widgets
class ScaledImage(QGraphicsView):
    """Like a QLabel or QGraphicsView but it allows you to scale the picture and maintains the aspect ratio"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.m_pixmapItem = self.scene().addPixmap(QPixmap())
        self.setAlignment(qt.Qt.AlignmentFlag.AlignCenter)

    @property
    def pixmap(self):
        return self.m_pixmapItem.pixmap()

    @pixmap.setter
    def pixmap(self, newPixmap):
        self.m_pixmapItem.setPixmap(newPixmap)
        self.fitInView(self.m_pixmapItem, qt.Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.m_pixmapItem, qt.Qt.AspectRatioMode.KeepAspectRatio)


# From this: https://stackoverflow.com/questions/74095602/how-to-adjust-video-to-widget-size-in-qt-python
class SquareVideo(QFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.background = QGraphicsView(self)
        self.background.setBackgroundBrush(qt.Qt.GlobalColor.black)
        self.background.setFrameShape(QFrame.Shape.NoFrame) # no borders
        self.background.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.background.setVerticalScrollBarPolicy(qt.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scene = QGraphicsScene()
        self.background.setScene(scene)
        self.videoItem = QGraphicsVideoItem()
        scene.addItem(self.videoItem)

        self.videoItem.nativeSizeChanged.connect(self.resizeBackground)

    def resizeBackground(self):
        # ensure that the view is always below any other child
        self.background.lower()
        # make the view as big as the parent
        self.background.resize(self.size())
        # resize the item to the video size
        self.videoItem.setSize(self.videoItem.nativeSize())
        # fit the whole viewable area to the item and crop exceeding margins
        self.background.fitInView(
            self.videoItem, qt.Qt.AspectRatioMode.KeepAspectRatioByExpanding)
        # scroll the view to the center of the item
        self.background.centerOn(self.videoItem)

    def showEvent(self, event):
        super().showEvent(event)
        self.resizeBackground()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # I feel like this would trigger recursion but I guess not
        self.resize(event.size().height(), event.size().height())

        self.resizeBackground()


class CameraWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.capture_handler = None
        self.last_hooptie = None
        self.active = True 

        self.layout = QVBoxLayout()

        self.viewfinder = SquareVideo()
        self.layout.addWidget(self.viewfinder)

        self.button = QPushButton(self)
        self.button.setText("RATE MY HOOPTIE!!!!!")
        self.layout.addWidget(self.button)

        self.wnd = QWidget(self)
        self.wnd.setLayout(self.layout)
        self.setLayout(self.layout)

        self.camera = QCamera()
        self.captureSession = QMediaCaptureSession()
        self.img_capture = QImageCapture(self.camera)

        self.captureSession.setCamera(self.camera)
        self.captureSession.setVideoOutput(self.viewfinder.videoItem)
        self.captureSession.setImageCapture(self.img_capture)

        self.button.clicked.connect(self.capture_hooptie)
        self.img_capture.errorOccurred.connect(self.on_error)
        self.camera.errorOccurred.connect(self.on_cam_error)
        self.img_capture.imageCaptured.connect(self.captured)

        self.camera.start()

    def set_capture_handler(self, handler):
        self.capture_handler = handler

    def capture_hooptie(self):
        assert self.img_capture.isReadyForCapture()
        self.img_capture.capture()

    def on_error(self, _id, _error, error_str):
        print(f"Error: {error_str}")

    def on_cam_error(self, _error, error_str):
        print(f"Error: {error_str}")

    def captured(self, id: int, img: QImage):
        assert self.capture_handler is not None
        self.capture_handler(img)


def speak(msg: str) -> sp.Popen:
            return sp.Popen(["say", "-v", "Daniel", "-r", "175", msg])


class RatingWidget(QWidget):
    def __init__(self, output_dir: pathlib.Path, silent: bool = False):
        super().__init__()
        self.critic = None
        self.output_dir = output_dir
        self.speech_proc = None
        self.silent = silent

        self.restart_handler = None

        self.review = None
        self.hooptie = None

        self.configure_view_layout()
        self.configure_control_layout()

        top_layout = QVBoxLayout()
        top_layout.addLayout(self.view_layout)
        top_layout.addLayout(self.control_layout)
        
        self.setLayout(top_layout)

    def configure_view_layout(self):
        self.view_layout = QHBoxLayout()

        self.hooptie_view = ScaledImage()
        self.view_layout.addWidget(self.hooptie_view, stretch=60)

        roast_font = QFont()
        roast_font.setPointSize(24)

        self.roast_view = QLabel()
        self.roast_view.setWordWrap(True)
        self.roast_view.setFont(roast_font)

        self.view_layout.addWidget(self.roast_view, stretch=40)

    def configure_control_layout(self):
        self.control_layout = QHBoxLayout()

        button_font = QFont()
        button_font.setPointSize(18)

        self.restart_button = QPushButton(self)
        self.restart_button.clicked.connect(self.restart)
        self.restart_button.setText("Rate another hooptie!")
        self.restart_button.setFont(button_font)
        self.control_layout.addWidget(self.restart_button)

        self.repeat_button = QPushButton(self)
        self.repeat_button.clicked.connect(self.say_again)
        self.repeat_button.setText("Roast me again!")
        self.repeat_button.setFont(button_font)
        self.control_layout.addWidget(self.repeat_button)

        self.save_button = QPushButton(self)
        self.save_button.clicked.connect(self.save_rating)
        self.save_button.setText("Save that shit!")
        self.save_button.setFont(button_font)
        self.control_layout.addWidget(self.save_button)

    def save_rating(self):
        if self.hooptie is None or self.review is None:
            button = QMessageBox.warning(self, "noHooptieWarning", "You must rate a hooptie first")
        else:
            hooptie_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save that shit!",
                    str(self.output_dir),
                    "Images (*.png)",
            ) 
            hooptie_path = pathlib.Path(hooptie_path)
            if hooptie_path.suffix == '':
                hooptie_path = hooptie_path.with_suffix(".png")
            
            review_path = hooptie_path.with_suffix(".txt")

            self.hooptie.save(str(hooptie_path))
            with open(review_path, 'w') as f:
                f.write(self.review)

    def set_restart_handler(self, handler):
        self.restart_handler = handler

    def restart(self):
        if self.speech_proc is not None:
            self.speech_proc.kill()
            self.speech_proc = None
            
        if self.restart_handler is not None:
            self.restart_handler()

    def say_again(self):
        if self.speech_proc is not None:
            self.speech_proc.kill()

        self.speech_proc = speak(self.review)

    def set_hooptie(self, img: QImage): 
        width = img.size().width()
        height = img.size().height()

        # We know that the image is always wider than it is tall (because it's from a wide-screen camera)
        left = int((width - height) / 2)
        self.hooptie = img.copy(left, 0, height, height)

        self.hooptie_view.pixmap = QPixmap.fromImage(self.hooptie)

        # Convert from QImage to PIL image
        img_buf = qt.QBuffer()
        img_buf.open(qt.QIODevice.OpenModeFlag.ReadWrite)
        self.hooptie.save(img_buf, "PNG")
        self.review = self.critic.rate(pil.Image.open(io.BytesIO(img_buf.data())))
        img_buf.close()

        self.roast_view.setText(self.review)

        # Say it out loud!
        if not self.silent:
            self.speech_proc = speak(self.review)
    
    def set_critic(self, critic: Critic):
        self.critic = critic


class MainWindow(QMainWindow):
    def __init__(
            self,
            output_dir: pathlib.Path,
            replay: bool = False,
            silent: bool = False
    ):
        super().__init__()
        self.layout = QStackedLayout()
        if output_dir is None:
            self.output_dir = pathlib.Path.home()
        else:
            self.output_dir = output_dir

        if not replay:
            self.camera_view = CameraWidget()
            self.camera_view.set_capture_handler(self.show_rating)
            self.layout.addWidget(self.camera_view)
        
        self.rating_view = RatingWidget(self.output_dir, silent)

        if replay:
            self.rating_view.set_restart_handler(self.load_from_file)
        else:
            critic = OpenaiCritic()
            self.rating_view.set_critic(critic)
            self.rating_view.set_restart_handler(self.show_camera)

        self.layout.addWidget(self.rating_view)

        self.layout.setCurrentIndex(0)

        window = QWidget()
        window.setLayout(self.layout)
        self.setCentralWidget(window)

    def show_rating(self, img: QImage):
        self.rating_view.set_hooptie(img)
        self.layout.setCurrentIndex(1)

    def show_camera(self):
        self.layout.setCurrentIndex(0)

    def load_from_file(self):
        hooptie_path, _ = QFileDialog.getOpenFileName(
                self,
                "Choose Your Hooptie",
                str(self.output_dir),
                "Images (*.png)",
        )

        if hooptie_path == "":
            return
        else:
            hooptie_path = pathlib.Path(hooptie_path)
            hooptie = QImage(str(hooptie_path))

            rating_path = hooptie_path.with_suffix(".txt")
            if rating_path.exists():
                critic = ReplayCritic(rating_path)
            else:
                critic = OpenaiCritic()

            self.rating_view.set_critic(critic)
            self.rating_view.set_hooptie(hooptie)


def main():
    parser = argparse.ArgumentParser(description="Rate your hooptie with the power of ARTIFICIAL INTELLIGENCE!")
    parser.add_argument("-r","--replay", action="store_true", help="Run in replay mode where you can load images from a file instead of the camera. If there is a '.txt' version of the image as well, then that will be displayed instead of using the AI critic.")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Save the images and predictions to the specified folder.")
    parser.add_argument("-s", "--silent", action="store_true", help="Supress the totally badass robot voice.")

    args = parser.parse_args()

    app = QApplication([])

    window = MainWindow(args.output, args.replay, args.silent)
    window.show()

    return app.exec()


if __name__ == '__main__':
    main()

