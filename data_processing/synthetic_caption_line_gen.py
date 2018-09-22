import os, StringIO
import re
import codecs
import subprocess
import glob
import cv2
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
from PIL import Image, ImageDraw,ImageFont
import cairo
from smi_extract import Subtitle, Smi
import ctypes as ct
import cairo
import random
import math
import pickle
import requests
from sys import stdout
import chardet
import sys
import shutil

font_familes ={
   'es': ["arial", "serif", "sans-serif", "cursive", "monospace", "times new roman", "american typewriter"],
   'en': ["arial", "serif", "sans-serif", "cursive", "monospace", "times new roman", "american typewriter"],
   'fr': ["arial", "serif", "sans-serif", "cursive", "monospace", "times new roman", "american typewriter"],
   'it': ["arial", "serif", "sans-serif", "cursive", "monospace", "times new roman", "american typewriter"],
   'de': ["arial", "serif", "sans-serif", "cursive", "monospace", "times new roman", "american typewriter"],
   'ja': ["arial", "serif", "sans-serif", "cursive", "monospace", "times new roman", "american typewriter"]
}



def convert_smi_to_txt(caption_content,output_path):
    smi = Smi(caption_content)
    frame_text = smi.convert('frame', lang='ENCC')
    lang = smi.detectlang()

    if len(frame_text) == 0:
        raise ValueError('no valid caption line in this file')

    # # convert timestamp(msec) to frame number
    # for caption in frame_text:
    #     caption[0] = int(caption[0] / 1000.0 * fps)
    #     caption[1] = int(caption[1] / 1000.0 * fps)

    assert len(frame_text) >= 100, "warning, subtitiles is not enough for generating!"

    starts = [x[0] for x in frame_text]
    ends = [x[1] for x in frame_text]
    print "totoal captions:", len(frame_text)  # ,frame_text[:]
    with open(output_path+'/caption_{}.txt'.format(lang),'w') as f:
        f.write(frame_text)
    f.close()
    return frame_text,lang



def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability: break
    return item



def text_extent(font_face, font_size, text, *args, **kwargs):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 0, 0)
    ctx = cairo.Context(surface)
    ctx.select_font_face(font_face, *args, **kwargs)

    # ctx.set_font_face(font_face)
    ctx.set_font_size(font_size)
    return ctx.text_extents(text)



def  generate_caption_png(caption,lang,i,path):
    #####set the 'caption style' parameters for each line####
    font_family = font_familes[lang]
    font = random.choice(font_family)
    stroke_color = random_pick([(1.0, 1.0, 1.0), (1.0, 1.0, 0)], [0.9, .1])
    stroke_style = random_pick([cairo.FONT_SLANT_NORMAL, cairo.FONT_SLANT_ITALIC, cairo.FONT_SLANT_OBLIQUE],
                               [0.6, 0.2, 0.2])
    stroke_width = cairo.FONT_WEIGHT_BOLD
    contour_width = math.fabs(random.gauss(0.4, 0.2))
    font_size = random.gauss(25, 5)



    font_args = [stroke_style,stroke_width]
    surface = cairo.ImageSurface.create_from_png(buffer)

    ctx = cairo.Context(surface)
    ctx.select_font_face(font, *font_args)
    ctx.set_font_size(font_size)
    # title="this is a \n test!"
    lines = caption.split('\n')
    n_line = len(lines)
    boxes = []
    for j, line in enumerate(lines):
        (x_bearing, y_bearing, text_w, text_h,
         x_advance, y_advance) = text_extent(font, font_size, line, *font_args)
        # print "x_bearing:{}, y_bearing:{}, text_w:{}, text_h:{},x_advance:{}, y_advance:{}".format(x_bearing, y_bearing, text_w, text_h,
        #  x_advance, y_advance)

        #
        # if location == 'bt':
        #     ctx.move_to((w - text_w) // 2, h - (n_line-i) * text_h*line_width)
        #     boxes.append([(w - text_w) // 2, h - (n_line-i) * text_h*line_width,text_w, text_h])
        # elif location == 'tp':
        #     ctx.move_to((w - text_w) // 2, i * text_h * line_width)
        #     boxes.append([(w - text_w) // 2, i * text_h * line_width,text_w, text_h])
        # else:
        #     raise RuntimeError("undefined location, currently only support 'bt' and 'tp'.")
        # ctx.text_path(title)
        ctx.set_source_rgb(stroke_color[0], stroke_color[1], stroke_color[2])
        ctx.text_path(line)
        ctx.fill_preserve()
        ctx.set_source_rgb(0.0, 0.0, 0.0)
        ctx.set_line_width(contour_width)
        ctx.stroke()
        surface.write_to_png(path + '/{}_{}_{}.png'.format(lang,i,j))
        with open(path + '/gt_{}_{}_{}.txt'.format(lang,i,j)) as f:
            f.write(line)
        f.close()
    # if len(boxes)>0:
    #     with open(path + '.pkl', 'wb') as f_bb:
    #         pickle.dump(boxes, f_bb)
    # pangocairo_context = pangocairo.CairoContext(context)
    # pangocairo_context.set_antialias(cairo.ANTIALIAS_SUBPIXEL)
    #
    # layout = pangocairo_context.create_layout()
    # fontname = sys.argv[1] if len(sys.argv) >= 2 else "Sans"
    # font = pango.FontDescription(fontname + " 25")
    # layout.set_font_description(font)
    #
    # layout.set_text(u"Hello World")
    # context.set_source_rgb(0, 0, 0)
    # pangocairo_context.update_layout(layout)
    # pangocairo_context.show_layout(layout)

    # font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 15)
    # text_w, text_h = draw.textsize(title, font)
    # draw.text(((w - text_w) // 2, h - text_h), title, (255,255,255), font=font)
    # # draw.text((REQ_WIDTH, REQ_HEIGHT), title, (255,255,255), font=font)
    #
    # img.save(path)

    return


if __name__ == '__main__':
    for lang in ['en','ja','es','fr','de','it']:
        print "processing language {}".format(lang)
        path_file = glob.glob('../data/{}_video_caption_*'.format(lang))
        print path_file
        with open(path_file[0],'rb') as f:
            data_paths = pickle.load(f)
        print len(data_paths)
        cnt = 0
        for [video_path, caption_path] in data_paths:
            filename = video_path.split('/')[-1].split('.')[0]
            output_path = '../data/{}/'.format(lang) + filename + '/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            # else:
            #     shutil.rmtree(output_path)

            print "processing {}th frame, name:{}...".format(cnt,filename)
            r_c = requests.get(caption_path, allow_redirects=True)
            caption_content = r_c.content
            # print caption_content
            encode_format = chardet.detect(caption_content)['encoding']
            print encode_format
            caption_content = caption_content.decode(encode_format)  # .decode('utf-16')#.decode('latin-1')
            print caption_content
            frame_text,lang_det = convert_smi_to_txt(caption_content,output_path)
            if lang !=lang_det:

                print "language type not agree, need to check!!!!"
                with open('languge_disagree.txt','w+') as f:
                    f.write(output_path)
                f.close()
            for i, caption in enumerate(frame_text):
                generate_caption_png(caption,lang_det,i,output_path)

            # add_caption_to_frame(video_path, caption_content, output_path)
            cnt +=1
        print "done!"