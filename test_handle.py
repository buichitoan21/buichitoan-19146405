# Import required Libraries
from tkinter import *
from PIL import Image, ImageTk
import cv2
import tkinter as tk
from tkvideo import tkvideo
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import os
from keras_preprocessing import image

# Create an instance of TKinter Window or frame
win = Tk()

# Set the size of the window
win.geometry("800x600")
win.title("NHAN DIEN KHUON MAT")
label =Label(win)
label.grid(row=0, column=0)
tit=Label(win,text='NHẬN DIỆN VÀ DỰ ĐOÁN ĐỘ TUỔI, GIỚI TÍNH, HÌNH DÁNH KHUÔN MẶT',bd=3,bg='white',fg='red',font=("Arial",16,"bold"))
tit.place(x=40,y=10)
note = Label(win, text="*chú ý: nhấn 'q' để thoát camera",font=("Arial",8,"bold"))
note.place(x=40,y=505)


def show_frames():

    btn_begin = tk.Button(text = 'Begin',bg="#78BCC4",font=("Arial",14,"bold"), command = btn_begin_click)
    btn_exit =  tk.Button(text = 'Exit' ,bg="#816AD6",font=("Arial",14,"bold"), command=lambda: win.quit())

    btn_begin.place(x = 240, y = 540)
    btn_exit.place(x = 440, y = 540)

def btn_begin_click():
    vid = cv2.VideoCapture(0)
    i=0

    face_cascade_name = 'haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier()

    #-- 1. Load the cascades
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)


    classes_age = ['1-3','4-9','10-18','19-21','22-27','28-33','34-45','46-56']
    new_model_age = load_model('model/model_age_10.h5')
    classes_shape = ['heart','oblong','oval','round','square'] 
    new_model_shape = load_model('model/model_shape_face_1.h5')
    classes_gender = ['female','male']
    new_model_gender = load_model('model/model_gender_1.h5')

    while(True):
        r, frame = vid.read()
        # cv2.imshow('frame', frame)
        cv2.imwrite('final' + str(i) + ".jpg", frame) 
        img = cv2.imread('final' + str(i) + ".jpg")
        test_image_1 = image.load_img('final' + str(i) + ".jpg", target_size=(150, 150))
        test_image_1 = image.img_to_array(test_image_1)
        test_image_1 = np.expand_dims(test_image_1, axis=0)
        face_cascade_name = 'haarcascade_frontalface_alt.xml'
        face_cascade = cv2.CascadeClassifier()
        if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
            print('--(!)Error loading face cascade')
            exit(0)
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        faces = face_cascade.detectMultiScale(frame_gray)
    
        for (x,y,w,h) in faces:
            # center = (x + w//2, y + h//2)
            frame = cv2.rectangle(img, (x,y), (x+w, y+h),(255,0,255),4)
            img_gray=frame_gray[y:y+h,x:x+w]
            

            cv2.imwrite('final' + str(i+1) + ".jpg", img_gray) 
            test_image = image.load_img('final' + str(i) + ".jpg", target_size=(150, 150))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)

            result = new_model_gender.predict(test_image_1)
            result1 = result[0]
            for w in range(2):
                if result1[w] == 1.:
                    break
            prediction_gender = classes_gender[w]
            position_gender = (x-120,y)

            result2 = new_model_shape.predict(test_image)
            result3 = result2[0]
            for e in range(5):
                if result3[e] == 1.:
                    break
            prediction_shape = classes_shape[e]
            position_shape = (x-120,y+30)

            result4 = new_model_age.predict(test_image)
            result5 = result4[0]
            for r in range(8):
                if result5[r] == 1.:
                    break
            prediction_age = classes_age[r]
            position_age = (x-120,y+60)


            cv2.putText(frame,"gender:",(x-240,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            cv2.putText(frame,prediction_gender,position_gender,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            cv2.putText(frame,"shape:",(x-230,y+30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            cv2.putText(frame,prediction_shape,position_shape,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            cv2.putText(frame,"age:",(x-230,y+60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            cv2.putText(frame,prediction_age,position_age,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            
            os.remove('final' + str(i+1) + ".jpg")
        
        cv2.imshow('age',frame)

        cv2image= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image = img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        label.after(20, show_frames)
        label.place(x = 70, y = 20)

        os.remove('final' + str(i) + ".jpg")
        i = i + 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()

def btn_end_click():
    cv2.destroyAllWindows()


show_frames()
print('ok')
win.mainloop()

