import cv2
import numpy as np
import pyscreenshot as ImageGrab
from matplotlib import pyplot as plt
import pyautogui
import random
import time
from PIL import Image


def screen_compatible(region):
    x1 = region[0]
    y1 = region[1]
    width = region[2]-x1
    height = region[3]-y1

    return pyautogui.screenshot(region=(x1,y1,width,height))

def imagesearch_small(image, tolerance = 0.8):
    return imagesearcharea(image,0,0,1366,768, tolerance)



def imagesearcharea(image, x1,y1,x2,y2, precision=0.8, im=None) :
    if im is None :
        im = screen_compatible(region=(x1, y1, x2, y2))
        #im.save('testarea.png')

    img_rgb = np.array(im)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(image, 0)
    #w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if (max_val < precision):
        return [-1, -1]
    return max_loc


def imagesearcharea_click(image, x1,y1, x2, y2, origin, offset, action, time):
    pos = imagesearcharea(image, x1, y1,x2, y2)
    pos = [pos[0] + origin[0] - offset, pos[1] + origin[1] - offset]
    if pos[0] == -1:
        return [-1]

    img = cv2.imread(image)
    height, width, channels = img.shape

    pyautogui.moveTo(pos[0] + width / 2 + 5 * random.random(), pos[1] + height / 2 + 5 * random.random(),
                     time)
    pyautogui.click(button=action)

    return pos

def imagesearch_click(image, action,time):
    max_loc = imagesearch(image)
    if (max_loc[0] == -1):
        return [-1]
    img= cv2.imread(image)
    height, width, channels = img.shape
    pyautogui.moveTo(max_loc[0]+width/2+5*random.random(), max_loc[1]+height/2+5*random.random(),time*random.random())
    pyautogui.click(button=action)
    return max_loc

def click_image(image,pos,  action, time):
    img = cv2.imread(image)
    height, width, channels = img.shape
    pyautogui.moveTo(pos[0] + width / 2 + 5 * random.random(), pos[1] + height / 2 + 5 * random.random(),
                     time * random.random())
    pyautogui.click(button=action)

def imagesearch(image, tolerance=0.8):
    im = pyautogui.screenshot()
    #im.save('testarea.png')
    img_rgb = np.array(im)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(image, 0)
    template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if (max_val < tolerance):
        return [-1,-1]
    return max_loc




def imagesearch_closest(image, pos):
    im = pyautogui.screenshot()
    img_rgb = np.array(im)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(image, 0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where(res >= threshold)
    mindelt = 9999999

    endpos = [-1,-1]
    for pt in zip(*loc[::-1]):
        delt = abs(pt[0] - pos[0])+ abs(pt[1] - pos[1])
        if (delt < mindelt):
            mindelt = delt
            endpos = pt

    return endpos

def imagesearch_no_screen(image, template_img):
    img_rgb = cv2.imread(image)

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_img, 0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if (max_val < 0.67):
        return [-1, -1]
    return max_loc

def imagesearch_loop(image):
    door_pos = imagesearch_small(image)
    i = 0
    while door_pos[0] == -1:
        print(image+" not found, waiting")
        time.sleep(r(0.5, 0.5))
        door_pos = imagesearch_small(image)
    return door_pos

def waitforimage(image, timesample):
    pos = imagesearch(image)

    while(pos[0] == -1):
        time.sleep(timesample)
        pos = imagesearch(image)
    return pos

def waitforimage_region(image, timesample, x1,y1,x2,y2):
    pos = imagesearcharea(image, x1,y1,x2,y2)

    while(pos[0] == -1):
        time.sleep(timesample)
        pos = imagesearcharea(image, x1, y1, x2, y2)
    return pos

# made to compare two images
# input : array containing the pixels (rgb)
def imagecompare(img_rgb, template, tolerance=0.8):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val < tolerance:
        return False
    else:
        return True

    return max_val

def r(num, rand):
    return num + rand*random.random()