from PIL import ImageTk
import cv2
from tkinter import *
from PIL import Image

width, height = 480, 360
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = Tk()
root.bind('<Escape>', lambda e: root.quit())
load = Image.open('test.jpg')
render = ImageTk.PhotoImage(load)

img = Label(root, image=render)
img.image = render
img.place(x=0, y=30)

lmain = Label(root)
lmain.pack()

root.geometry("1000x800+300+300")

root.mainloop()