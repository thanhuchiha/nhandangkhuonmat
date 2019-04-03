# https://phocode.com/blog/2016/01/31/tkinter-quan-ly-layout/
#real time client server
import PIL
import cv2
from tkinter import *
from PIL import Image, ImageTk

import os

# Kích thước camera và xử lý

width, height = 480, 900
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = Tk()
root.bind('<Escape>', lambda e: root.quit())
root.configure(background="#232528")

#Viet tren day -> Bo anh len day chay
#Show Image
i = 1
j = 0
"""while i <= 6:
    size = 120, 150
    load = Image.open('dataset/thanhuchiha/1.'+str(i)+'.jpg')
    load.thumbnail(size, Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)

    img = Label(root, image=render,bd=1, relief="solid")
    img.image = render

    img.place(x=5, y=10+j)
    j = j + 120
    i = i + 1
i = 1
j = 0
while i <= 6:
    size = 120, 150
    load = Image.open('dataset/thanhuchiha/1.'+str(i)+'.jpg')
    load.thumbnail(size, Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)

    img = Label(root, image=render,bd=1, relief="solid")
    img.image = render
    img.place(x=130, y=10+j)
    j = j + 120
    i = i + 1"""


lmain = Label(root, bd=1, relief="solid", width=600, height=480, padx=3)
lmain.pack()


# Hàm show camera
def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    #frame = cv2.resize(frame, (480, 360))
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)

    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)



class Example(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.parent = parent
        self.initUI()

    def initUI(self):
        self.parent.title("Face Recognition")
        self.pack(fill=BOTH, expand=True)
        self.var = BooleanVar()
        self.configure(background="#232528")

        # Viết các thuộc tính trong đây

        btn = Button(self, text="Điểm danh", bg="#06b1e5",fg="blue", font="Times 16 bold", command=self.onClick())
        btn.place(x=530, y=20)

        # Hiển thị hình ảnh và độ tin cậy

        # load = Image.open(file='chat.jpg')
        # render = ImageTk.PhotoImage(load)

        # img = Label(self, image=render)
        # img.image = render
        # img.place(x=0, y=0)

    # Hàm xử lý sự kiện cho các nút
    def onClick(self):
        if self.var.get() == True:
            self.master.title("Checkbutton")
        else:
            self.master.title("")


#os.system('python recognize_video.py')
show_frame()
#app = Example(root)
root.geometry("1200x550+300+300")

root.mainloop()
