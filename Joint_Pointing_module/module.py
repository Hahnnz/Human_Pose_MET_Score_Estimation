import tkinter, PIL, os, glob, math
import numpy as np
from PIL import ImageTk, Image
from tkinter import ttk

Dataset_root="./image"

def explore_dir(dir,count=0,f_extensions=None):
    if count==0:
        global n_dir, n_file, filenames, filelocations
        n_dir=n_file=0
        filenames=list()
        filelocations=list()

    for img_path in sorted(glob.glob(os.path.join(dir,'*' if f_extensions is None else '*.'+f_extensions))):
        if os.path.isdir(img_path):
            n_dir +=1
            explore_dir(img_path,count+1)
        elif os.path.isfile(img_path):
            n_file += 1
            filelocations.append(img_path)
            filenames.append(img_path.split("/")[-1])
    return np.array((filenames,filelocations))

def create_canvas():

	def joint_pointing_canvas(canvas,imgimg_name):
		canvas.delete("all")
		resized=False

		img = Image.open(imgimg_name)
		if hasattr(img, '_getexif') and 'jpeg' in imgimg_name:
		    orientation = 0x0112
		    exif = img._getexif()
		    if exif is not None:
		        orientation = exif[orientation]
		        rotations = {
		            3: Image.ROTATE_180,
		            6: Image.ROTATE_270,
		            8: Image.ROTATE_90
		        }
		        if orientation in rotations:
		            img = img.transpose(rotations[orientation])
		OLD_W, OLD_H = img.size
		NEW_W, NEW_H = 800, 600

		if OLD_W>NEW_W or OLD_H>NEW_H: 
			img = img.resize((NEW_W, NEW_H), Image.ANTIALIAS)
			resized=True

		canvas = tkinter.Canvas(canvas, width=NEW_W, heigh=NEW_H)
		canvas.place(x=0,y=0)

		img = ImageTk.PhotoImage(img)
		canvas.create_image(0,0,image=img,anchor="nw")
		filefullpath.delete(0,"end")
		filefullpath.insert("end",imgimg_name)
		coor_entry.delete(0,"end")

		def down(event):
			global x0,y0;
			x0, y0 = event.x, event.y	  
			if(x0,y0) == (event.x,event.y):
				canvas.create_oval(x0-2,y0-2,event.x+2,event.y+2,outline="red",fill="red",width=2)
		def up(event):
			global x0, y0
			if(x0,y0) == (event.x,event.y):
				canvas.create_oval(x0-2,y0-2,event.x+2,event.y+2,outline="red",fill="red",width=2)
			if resized : 
				x0*=OLD_W/NEW_W
				y0*=OLD_H/NEW_H
			coor_entry.insert("end",str(int(x0))+","+str(int(y0))+",")

		canvas.bind("<Button-1>",down) 
		canvas.bind("<ButtonRelease>",up) 
		root.mainloop()

	def create_button(class_name,cur_class):
		class_name.config(text="Current Class : "+cur_class, font='Helvetica 18 bold')
		for i in range(len(img_paths_with_class[classes.index(cur_class)])):
			tkinter.Button(img_button_frame, text=img_paths_with_class[classes.index(cur_class)][i].split("/")[-1], width=5,
				command=lambda i=i: joint_pointing_canvas(canvas,img_paths_with_class[classes.index(cur_class)][i])).grid(column=int(i/6),row=int(i%6))

	dataset=explore_dir(Dataset_root,0)
	img_paths = dataset[1]
	classes=sorted(set(img_paths[i].split("/")[-2] for i in range(len(img_paths))))
	img_paths_with_class=list([list() for _ in range(len(set(img_paths)))])
	
	for i, Class in enumerate(classes):
	    for path in img_paths:
	        if Class in path.split("/")[-2] : img_paths_with_class[i].append(path)

	max_size=np.array([0,0])

	root = tkinter.Tk()
	root.title("Human Joint Point [Version 3]")

	img_button_frame=tkinter.Frame(root)
	canvasframe=tkinter.Frame(root)
	option_button_frame=tkinter.Frame(root)
	class_button_frame=tkinter.Frame(root)
	cur_class_frame=tkinter.Frame(root)

	class_button_frame.pack(ipadx=0, ipady=0, side="left")
	option_button_frame.pack(ipadx=0, ipady=0, side="left")
	img_button_frame.pack(ipadx=0, ipady=0, side="top")
	cur_class_frame.pack(ipadx=0, ipady=0, side="top")
	tkinter.Label(class_button_frame,text="Classes", font='Helvetica 18 bold').pack()

	cur_class_name = tkinter.Label(cur_class_frame)
	cur_class_name.pack()
	coor=tkinter.StringVar(root,value='')
	coor_entry=ttk.Entry(root,textvariable=coor,width=100)
	coor_entry.pack(side="top")

	full_path=tkinter.StringVar(root,value='')
	filefullpath=ttk.Entry(root,textvariable=full_path,width=80)
	filefullpath.pack(side="top")

	canvasframe.pack(ipadx=0,ipady=0,side="top")
	canvas = tkinter.Canvas(canvasframe,width=800,heigh=600)
	canvas.pack()

	for i, name in enumerate(classes):
		tkinter.Button(class_button_frame, text=name, width=15, command=lambda i=i: create_button(cur_class_name,classes[i])).pack()

	tkinter.Button(option_button_frame, text="Absence", width=7, command=lambda : coor_entry.insert("end","-1,-1,")).pack()
	tkinter.Button(option_button_frame, text="Visible", width=7, command=lambda : coor_entry.insert("end","0,")).pack()
	tkinter.Button(option_button_frame, text="Unvisible", width=7, command=lambda : coor_entry.insert("end","1,")).pack()

	copyright_frame=tkinter.Frame(root)
	copyright_frame.pack(side="bottom")
	tkinter.Label(copyright_frame,
	 text="⊙ Copyrightⓒ2018 by Hahnnz, Brain Information Laboratory @ Incheon National University(INU). All rights reserved.").pack()

	root.mainloop()

if __name__ == '__main__':
    create_canvas()
