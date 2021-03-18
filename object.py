import cv2
import numpy as np
import smtplib
import time
import datetime
import pyautogui
now = datetime.datetime.now()
print ("Current date and time : ")
print (now.strftime("%S"))
# Load Yolo
def mytimer():
    if cfp=="Not wearing mask&helmet" or cfp=="Wearing Mask,no helmet" or cfp=="wearing helmet ,no mask":
        print("started messaging\n")
        try:
            s = smtplib.SMTP('smtp.gmail.com', 587)
            s.starttls()
            s.login("Facemaskdetector@gmail.com", "majorproject")
            SUBJECT = "mask dection project"
            TEXT = "person not wearing mask or helmet was dected in system."
            message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT)
            s.sendmail("Facemaskdetector@gmail.com", "apoorvamy85@gmail.com", message)
            s.quit()
            print("sucessfull")
            myScreenshot = pyautogui.screenshot()
            myScreenshot.save(r'screenname.png')
            print("saved screen shot")
        except:
            print("not sucessfull")



    else:
        print("mask and helmet was dected")
net = cv2.dnn.readNet("yolov3_training_final.weights", "yolov3_testing.cfg")
classes = []
with open("test1.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# Loading camera
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320,320), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

    for i in range(len(boxes)):
        if i in indexes:
            print(str(classes[class_ids[i]]))
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cfp=classes[class_ids[i]]
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)
            now = datetime.datetime.now()
            lk=now.strftime("%S")
            print(lk)
            if lk=="30" or lk =="15" or lk=="45" or lk=="59":
                mytimer()
            else:
                pass
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 9)), (10, 50), font, 3, (0, 0, 0), 3)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()