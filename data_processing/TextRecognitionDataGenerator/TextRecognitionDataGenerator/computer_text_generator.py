import cv2
import math
import random
import os
import numpy as np

from PIL import Image, ImageFont, ImageDraw, ImageFilter



class ComputerTextGenerator(object):
    @classmethod
    def generate(cls, text, font, text_color, height,contour_with):
        font_size = int(height)
        image_font = ImageFont.truetype(font=font, size=font_size)
        # print(font_size)
        text_width, text_height = image_font.getsize(text)

        txt_img = Image.new('RGBA', (text_width+contour_with*2, text_height+contour_with*2), (0, 0, 0, 0))

        txt_draw = ImageDraw.Draw(txt_img)

        # fill = random.randint(text_color[0], text_color[-1])
        fill = text_color
        shadowcolor = (0,0,0)
        x=contour_with
        y=contour_with

        txt_draw.text((x - contour_with, y -contour_with), text, font=image_font, fill=shadowcolor)
        txt_draw.text((x + contour_with, y - contour_with), text, font=image_font, fill=shadowcolor)
        txt_draw.text((x - contour_with, y + contour_with), text, font=image_font, fill=shadowcolor)
        txt_draw.text((x + contour_with, y + contour_with), text, font=image_font, fill=shadowcolor)
        txt_draw.text((x,y ), text, fill=fill, font=image_font)

        return txt_img
