#!/usr/bin/env python3
import pathlib
import cv2
import PIL as pil
from PIL import Image
import io
import base64
import textwrap
from openai import OpenAI
import subprocess as sp


def getPrediction(rawImage, client):
    im = pil.Image.fromarray(cv2.cvtColor(rawImage, cv2.COLOR_BGR2RGB))
    im = im.resize((512, 512))

    b = io.BytesIO()
    im.save(b, format="JPEG")

    completion = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is this person wearing a hat?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{im}",
                            "detail": "low"
                        },
                    }
                ],
            }
        ],
        max_tokens=200
    )

    return(completion.choices[0].message)


def applyMsg(img, msg):
    textImg = img.copy()
    height, width, channel = textImg.shape

    font = cv2.FONT_HERSHEY_SIMPLEX

    wrapped = textwrap.wrap(msg, width=45)
    x, y = 10, 40
    font_size = 1
    font_thickness = 2

    for i, line in enumerate(wrapped):
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]

        gap = textsize[1] + 10

        y = int((textImg.shape[0] + textsize[1]) / 2) + i * gap
        x = int((textImg.shape[1] - textsize[0]) / 2)

        cv2.putText(textImg, line, (x, y), font,
                    font_size,
                    (255,255,255),
                    font_thickness+1,
                    lineType = cv2.LINE_AA)

        cv2.putText(textImg, line, (x, y), font,
                    font_size,
                    (0,0,0),
                    font_thickness,
                    lineType = cv2.LINE_AA)

    return textImg


def main():
    # OAIclient = OpenAI()
    cam = cv2.VideoCapture(0)
    windowName = "Your Hooptie"
    cv2.namedWindow(windowName)

    while True:
        ret, frame = cam.read()
        if not ret:
            raise RuntimeError("Camera failed!")

        # Crop assuming the raw frame is wider than it is tall
        xMargin = int((frame.shape[1] - frame.shape[0]) / 2)
        frame = frame[:, xMargin:-xMargin]
        cv2.imshow(windowName, frame)
        k = cv2.waitKey(33)
        if k == 32:
            # msg = getPrediction(frame, client)
            msg = "Your racecar is bad and you should feel bad"

            textImg = applyMsg(frame, msg)
            cv2.imshow(windowName, textImg)

            speechProc = sp.Popen(["say", msg])

            cv2.waitKey(0)
            speechProc.wait()
        elif k == 27:
            break


main()
