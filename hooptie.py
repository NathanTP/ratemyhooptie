#!/usr/bin/env python3
import cv2
import PIL as pil
from PIL import Image
import io
import base64
import textwrap
from openai import OpenAI
import subprocess as sp
import random
import numpy as np


def getPrediction(rawImage, client):
    im = pil.Image.fromarray(cv2.cvtColor(rawImage, cv2.COLOR_BGR2RGB))
    im = im.resize((512, 512))

    b = io.BytesIO()
    im.save(b, format="JPEG")
    img_str = base64.b64encode(b.getvalue()).decode("utf-8")

    completion = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are an expert in automotive engineering. You have a sardonic personality and are prone to witty commentary. Your goal is to identify the automobile in this picture and then say something provacative and amusing about the automobile. You should make fun of its owners and question their judgment for bringing the car here. You should end with something amusing and cheeky. "},
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


def applyMsg(img, msg):
    textImg = img.copy()
    height, width, channel = textImg.shape

    font = cv2.FONT_HERSHEY_SIMPLEX

    wrapped = textwrap.wrap(msg, width=38)
    x, y = 10, 40
    font_size = 1
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


def display(img, msg, windowName):
    textBox = np.zeros([1080,675,3],dtype=np.uint8)
    textBox.fill(0)
    textBox = applyMsg(textBox, msg)

    imgBig = cv2.resize(img, (1080,1080))
    textImg = np.concatenate((imgBig, textBox), axis=1)

    k = 13
    while k == 13:
        cv2.imshow(windowName, textImg)
        speechProc = sp.Popen(["say", "-r", "210", msg])

        k = cv2.waitKey(0)
        speechProc.wait()


def main():
    OAIClient = OpenAI()
    cam = cv2.VideoCapture(0)
    imgWindow = "Your Hooptie"
    cv2.namedWindow(imgWindow, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cam.read()
        if not ret:
            raise RuntimeError("Camera failed!")

        # Crop assuming the raw frame is wider than it is tall
        xMargin = int((frame.shape[1] - frame.shape[0]) / 2)
        frame = frame[:, xMargin:-xMargin]
        cv2.imshow(imgWindow, frame)
        k = cv2.waitKey(33)
        if k == 32:
            # frame = cv2.imread("ratings/5578118396.png")
            name = str(random.randint(0, 10e9))
            msg = getPrediction(frame, OAIClient)
            # with open("ratings/5578118396.txt", 'r') as f:
            #     msg = f.read()
            # msg = "Your racecar is bad and you should feel bad"

            with open("ratings/" + name + ".txt", 'w') as f:
                f.write(msg)
            cv2.imwrite("ratings/" + name + ".png", frame)

            display(frame, msg, imgWindow)
        elif k == 27:
            break


main()

# OAIClient = OpenAI()
# Must be a square aspect-ratio image in jpeg format
# frame = cv2.imread("PATH/TO/IMAGE.jpeg")
# print(getPrediction(frame, OAIClient))
