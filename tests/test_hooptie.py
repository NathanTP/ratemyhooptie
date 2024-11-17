import pathlib
import numpy as np
import pytest

import cv2

import ratemyhooptie as ht


resources = pathlib.Path(__file__).parent / "testResources"


# from https://stackoverflow.com/questions/33084190/default-skip-test-unless-command-line-parameter-present-in-py-test
external_model = pytest.mark.skipif("not config.getoption('external_model')")


class MockCritic(ht.Critic):
    def __init__(self, msg: str):
        self.msg = msg

    def rate(img: np.ndarray) -> str:
        return self.msg


def test_crop():
    img = cv2.imread(str(resources / "test.png"))
    assert img is not None
    ht.crop_square(img)


def test_camera():
    cam = cv2.VideoCapture(0)
    ret, img = cam.read()
    assert ret


def test_apply_msg():
    msg = "foo bar"
    text_box = np.zeros([1920,1280,3],dtype=np.uint8)
    text_box.fill(0)
    text_box = ht.apply_msg(text_box, msg)
    assert text_box is not None


def test_replay_critic():
    critic = ht.ReplayCritic(resources / "test.txt")
    
    img = cv2.imread(str(resources / "test.png"))
    assert img is not None

    msg = critic.rate(img)

    with open(resources / 'test.txt', 'r') as f:
        real_msg = f.read()

    assert msg == real_msg


@external_model
def test_openai_critic():
    critic = ht.OpenaiCritic()

    img = cv2.imread(str(resources / "test.png"))
    assert img is not None

    msg = critic.rate(img)

    assert msg is not None
    assert len(msg) > 10

    print(f"Message was: {msg}")

