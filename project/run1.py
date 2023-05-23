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
camera = 0
weights = 'best_face.pt'
width, height = (352, 288)
display = False

bot = telepot.Bot(token)
bot.sendMessage(receiver_id, 'Your camera is active now.')

device = torch.device('cpu')

model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())
cudnn.benchmark = True

cap = cv2.VideoCapture(camera)
cap.set(3, width)
cap.set(4, height)

import torch.nn as nn

for m in model.modules():
    if isinstance(m, nn.Upsample):
        m.recompute_scale_factor = None
#Added code in updatedrun
sent_image = False  # flag variable to track if an image has been sent.The flag variable sent_image is initially set to False. Once an image is sent to the Telegram bot, the flag is set to True, and the loop breaks out immediately. This will terminate the program.
while(cap.isOpened() and not sent_image):
    time.sleep(0.2)
    ret, frame_ = cap.read()
    frame = cv2.resize(frame_, (width, height), interpolation = cv2.INTER_AREA)

    if ret and (frame.mean() < 10 or cv2.Laplacian(frame, cv2.CV_64F).var() < 50):
        bot.sendMessage(receiver_id, text='Camera lens is covered or damaged!')
    else:
        if ret == True:
            img = torch.from_numpy(frame).float().to(device).permute(2, 0, 1)
            img /= 255.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=False)[0]
            pred = non_max_suppression(pred, 0.39, 0.45, agnostic=True)

            for det in pred:
                if len(det):
                    conf_, class_ = det[0][4], int(det[0][5])

                    if class_ == 0  and conf_ > 0.35 :
                        time_stamp = int(time.time())
                        fcm_photo = f'detected/{time_stamp}.png'
                        cv2.imwrite(fcm_photo, frame_)
                        bot.sendPhoto(receiver_id, photo=open(fcm_photo, 'rb'))
                        print(f'{time_stamp}.png has sent.')
                        sent_image = True  #Added code to set flag to True after sending image
                        break  # break out of loop after sending image
                        time.sleep(1)
        else:
            break

cap.release()
cv2.destroyAllWindows()
