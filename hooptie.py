#!/usr/bin/env python
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
import argparse
import pathlib


usage = """
Usage:
    Press space to take a picture of your hooptie. The picture and the ARTIFICIAL INTELLIGENCEs rating will be displayed (probably). Press ENTER to replay the message or any other key to rate a new hooptie.

    Your hooptie and its rating will be saved for future enjoyment (use the --output argument to control where).
"""

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


def applyMsg(img, msg):
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


def display(img, msg, windowName, silent=False):
    """Display the image and message, and read the message. Press "enter" to
    replay, or any other key to exit."""
    # textBox = np.zeros([1080,675,3],dtype=np.uint8)
    textBox = np.zeros([1920,1280,3],dtype=np.uint8)
    textBox.fill(0)
    textBox = applyMsg(textBox, msg)

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


def cropSquare(img):
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


def main():
    parser = argparse.ArgumentParser(description="Rate your hooptie with the power of ARTIFICIAL INTELLIGENCE!")
    parser.add_argument("-f", "--file", type=pathlib.Path, help="Rate a static image rather than using the camera. If there is a FILENAME.txt in the same directory, that will be used as the rating rather than asking the ARTIFICIAL INTELLIGENCE!.")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Save the joint img+prediction to the specified file.")
    parser.add_argument("-s", "--silent", action="store_true", help="Supress the totally badass robot voice.")

    args = parser.parse_args()

    print(usage)

    imgWindow = "Your Hooptie"
    cv2.namedWindow(imgWindow, cv2.WINDOW_NORMAL)

    if args.file is not None:
        imgPath = args.file.resolve()
        msgPath = imgPath.with_suffix(".txt")

        img = cv2.imread(str(imgPath))
        img = cropSquare(img)

        if msgPath.exists():
            with open(msgPath, 'r') as f:
                msg = f.read()
        else:
            OAIClient = OpenAI()
            msg = getPrediction(img, OAIClient)

        msgImg = display(img, msg, imgWindow, args.silent)

        if args.output is not None:
            cv2.imwrite(str(args.output), msgImg)
    else:
        cam = cv2.VideoCapture(0)
        OAIClient = OpenAI()
        while True:
            ret, frame = cam.read()
            if not ret:
                raise RuntimeError("Camera failed!")

            frame = cropSquare(frame)
            cv2.imshow(imgWindow, frame)
            k = cv2.waitKey(33)
            if k == 32:
                name = str(random.randint(0, 10e9))
                msg = getPrediction(frame, OAIClient)

                outDir = "thill24_race1/"
                with open(outDir + name + ".txt", 'w') as f:
                    f.write(msg)
                cv2.imwrite(outDir + name + ".png", frame)

                msgImg = display(frame, msg, imgWindow, args.silent)
                if args.output is not None:
                    cv2.imwrite(str(args.output), msgImg)
            elif k == 27:
                break


if __name__ == "__main__":
    main()
