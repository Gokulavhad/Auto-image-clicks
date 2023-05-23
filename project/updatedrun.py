import time
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import non_max_suppression
import telepot


token = '6272699778:AAFb1GQ5gvoUvTsJTkyLq1FYbBvPEwW9WRc' # telegram token
receiver_id = 6024749639 # https://api.telegram.org/bot<TOKEN>/getUpdates
camera = 0 # webcam
weights = 'best_face.pt'
width, height = (352, 288) # quality 
display = False

bot = telepot.Bot(token)
bot.sendMessage(receiver_id, 'Your camera is active now.') # send a activation message to telegram receiver id

device = torch.device('cpu')

model = attempt_load(weights, map_location=device)  
stride = int(model.stride.max())  
cudnn.benchmark = True

cap = cv2.VideoCapture(camera)
cap.set(3, width) # width
cap.set(4, height) # height

import torch.nn as nn

for m in model.modules():
    if isinstance(m, nn.Upsample):
        m.recompute_scale_factor = None

while(cap.isOpened()):
    time.sleep(0.2) # wait for 0.2 second 
    ret, frame_ = cap.read()
    frame = cv2.resize(frame_, (width, height), interpolation = cv2.INTER_AREA)

    # Check if the camera lens is covered or damaged
    if ret and (frame.mean() < 10 or cv2.Laplacian(frame, cv2.CV_64F).var() < 50):
        # If the lens is covered or damaged, send a notification to the Telegram bot
        bot.sendMessage(receiver_id, text='Camera lens is covered or damaged!')
    else:
        if ret ==True:
            img = torch.from_numpy(frame).float().to(device).permute(2, 0, 1)
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=False)[0]
            pred = non_max_suppression(pred, 0.39, 0.45, agnostic=True) # img, conf, iou, ...

            for det in pred:
                if len(det):
                    conf_, class_ = det[0][4], int(det[0][5])

                    if class_ == 0  and conf_ > 0.35 :
                        time_stamp = int(time.time())
                        fcm_photo = f'detected/{time_stamp}.png'
                        cv2.imwrite(fcm_photo, frame_) # notification photo
                        bot.sendPhoto(receiver_id, photo=open(fcm_photo, 'rb')) # send message to telegram
                        print(f'{time_stamp}.png has sent.')
                        time.sleep(1) # wait for 1 second. Only when it detects.
        else:
            break
    
cap.release()
cv2.destroyAllWindows()
