import tkinter as tk
from tkinter import filedialog, Text
import tkinter.font as font
import os
import model_predict
import model_predict
root = tk.Tk()
root.title("Explicit Content Classifier")

myFont = font.Font(family='Playfair Display')

def openf():
    foldername = filedialog.askdirectory(initialdir='/', title='Select Folder')
    global directory
    directory = foldername
    
def predictions():
    for widget in frame.winfo_children():
        widget.destroy()
    
    cn, cs = model_predict.predict(directory)

    lab = tk.Label(frame, text = "Done!! \n")
    lab.config(font =("Playfair Display", 18))
    lab.pack()

    l1 = tk.Label(frame, text = "NSFW Files Count : " + str(cn) + '\n')
    l1.config(font =("Playfair Display", 18))
    l1.pack()

    l2 = tk.Label(frame, text = "SFW Files Count : " + str(cs))
    l2.config(font =("Playfair Display", 18))
    l2.pack()


canvas = tk.Canvas(root, height = 400, width = 800, bg = '#29465B')
canvas.pack()

frame = tk.Frame(root, bg = 'white')
frame.place(relwidth=0.8, relheight=0.7, relx=0.1, rely=0.1)

Folder = tk.Button(root, text = 'Choose Folder', padx = 10, pady = 5,
                    fg = 'white', bg = '#29465B', command = openf, width=27,
                    font = myFont)
Folder.pack(side = tk.LEFT)

predict = tk.Button(root, text = 'Predict', padx = 10, pady = 5,
                    fg = 'white', bg = '#29465B', command = predictions, width=27,
                    font = myFont)
predict.pack(side = tk.LEFT)

des = tk.Button(root, text = 'Exit', padx = 10, pady = 5,
                    fg = 'white', bg = '#29465B', command = root.destroy, width=27,
                    font = myFont)
des.pack(side = tk.LEFT)

root.mainloop()