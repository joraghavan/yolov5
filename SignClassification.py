import PySimpleGUI as sg
import torch
import detect as dt
import cv2
from PIL import Image

from tk import *


import pytesseract
img_cv = cv2.imread('../../images/signs2.png')
img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
img_cv = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
img_cv = cv2.bitwise_not(img_cv)

#cv2.imshow('', img_cv)
#cv2.waitKey(0)


#print(pytesseract.image_to_string(img_cv))

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

img = Image.open("../../images/signs2.png")
width = img.width
height = img.height
print()

dt.run(weights='yolov5s.pt',
        source='../../images',
        data='data/coco128.yaml',
        imgsz=(width, height),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_txt=True,
        save_conf=True,
        save_crop=False,
        nosave=False,
        classes=[2, 11],
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=True,
        project='results',
        name='processed',
        exist_ok=True,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,)

layout = [
    [sg.Image(filename="results/processed/signs2.png")]
]

window = sg.Window("Sign Classification", layout)




layout = [
    [
        sg.Graph(
            canvas_size=(width, height),
            graph_bottom_left=(0, 0),
            graph_top_right=(width, height),
            key="graph",
            enable_events=True,
            change_submits=True,
            drag_submits=True
        )
    ]
]

while True:
    event, values = window.read()

    if event == "Exit" or event == sg.WIN_CLOSED:
        break

window.close()




window = sg.Window("Sign Classification", layout)
window.Finalize()

graph = window.Element("graph")


graph.DrawImage(filename="../../images/signs2.png", location=(0, height))


f = open('results/processed/labels/signs2.txt', 'r')

for line in f.readlines():
    coordinates = line.split()
    sizeX = float(coordinates[3]) * width
    sizeY = float(coordinates[4]) * height
    x1 = float(coordinates[1]) * width - sizeX / 2
    y1 = (1 - float(coordinates[2])) * height - sizeY / 2
    y1Crop = float(coordinates[2]) * height - sizeY / 2
    x2 = x1 + sizeX
    y2 = y1 + sizeY
    y2Crop = y1Crop + sizeY

    if float(coordinates[0]) == 2:
        graph.DrawRectangle((x1, y1), (x2, y2), line_color="blue")
    elif float(coordinates[0]) == 11:
        graph.DrawRectangle((x1, y1), (x2, y2), line_color="red")



#0.685938 0.350939 0.121875 0.180751

#img_cropped = img.crop((x1, y1Crop, x2, y2Crop))
#img_cropped.show()

dragging = False
start_point = end_point  = None
iter = 0
while True:
    event, values = window.read()
    if event == "graph":
        #graph.DrawRectangle((x, y), (x+20, y+20), line_color="black")
        x, y = values["graph"]
        if not dragging:
            start_point = (x, y)
            dragging = True
            drag_figures = graph.get_figures_at_location((x, y))
            lastx = x
            lasty = y
        else:
            end_point = (x, y)
            dragging = False
            graph.DrawRectangle(start_point, end_point, line_color="black")
            iter += 1
            print(iter)
            #img_cropped = img_cv.crop(start_point, end_point)
            img_cropped = img_cv[start_point[0]:end_point[0], start_point[1]:end_point[1]]

            # img = Image.fromarray(img_cropped, 'RGB')
            # img.show()
            # printt(pytesseract.image_to_string(img_cropped))

        lastx = x
        lasty = y

    if event == "Exit" or event == sg.WIN_CLOSED:
        break

window.close()