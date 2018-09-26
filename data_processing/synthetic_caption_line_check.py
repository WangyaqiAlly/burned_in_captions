from __future__ import unicode_literals
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
from langdetect import detect
import sys
reload(sys)
sys.setdefaultencoding('utf8')

languages = ['neg','en','ja','es','fr','it','de']

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


    if len(frame_text) <100:
        raise ValueError("warning, subtitiles is not enough for generating!"
                         " only {} lines".format(len(frame_text)))
    rand_idx = random.sample(range(len(frame_text)), 100)
    langs =[0]*7
    most_lang = 0
    for idx in rand_idx:
        try:
            # print "detecting: {}".format( unicode(frame_text[idx][2]))
            lang_i = detect(frame_text[idx][2])
            # print lang_i
        except:
            continue
        if lang_i not in languages:
            continue
        i = languages.index(lang_i)
        langs[i] += 1
        if langs[i] > langs[most_lang]:
            most_lang = i
    lang =languages[most_lang]
    print "detected language:",lang


    # # convert timestamp(msec) to frame number
    # for caption in frame_text:
    # for caption in frame_text:
    #     caption[0] = int(caption[0] / 1000.0 * fps)
    #     caption[1] = int(caption[1] / 1000.0 * fps)



    # starts = [x[0] for x in frame_text]
    # ends = [x[1] for x in frame_text]
    print "totoal captions:", len(frame_text)  # ,frame_text[:]
    # with open(output_path+'/caption_{}.txt'.format(lang),'w') as f:
    #     f.write(frame_text)
    # f.close()
    cnt_line = 0
    with open(output_path + '/{}_captions.txt'.format(lang), 'wb') as f:
        for caption in frame_text:
            lines = caption[2].split('\n')
            n_line = len(lines)
            for line in lines:
                if len(line) < 1:
                    continue
                # print("writing line: {}".format(line))
                f.write(line+'\n')

                cnt_line += 1
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



def  generate_caption_png(line,lang,i,path):
    #####set the 'caption style' parameters for each line####
    font_family = font_familes[lang]
    font = random.choice(font_family)
    stroke_color = random_pick([(1.0, 1.0, 1.0), (1.0, 1.0, 0)], [0.9, .1])
    stroke_style = random_pick([cairo.FONT_SLANT_NORMAL, cairo.FONT_SLANT_ITALIC, cairo.FONT_SLANT_OBLIQUE],
                               [0.0, 1.0, 0.0])
    stroke_width = cairo.FONT_WEIGHT_BOLD
    contour_width = math.fabs(random.gauss(0.4, 0.2))
    font_size = random.gauss(40, 5)

    # font_family = font_familes[lang]
    # font = random.choice(font_family)
    # stroke_color = random_pick([(1.0, 1.0, 1.0), (1.0, 1.0, 0)], [0.9, .1])
    # stroke_style = random_pick([cairo.FONT_SLANT_NORMAL, cairo.FONT_SLANT_ITALIC, cairo.FONT_SLANT_OBLIQUE],
    #                            [0.6, 0.2, 0.2])
    # stroke_width = cairo.FONT_WEIGHT_BOLD
    # contour_width = math.fabs(random.gauss(0.4, 0.2))
    # font_size = random.gauss(25, 5)

    print font,stroke_color,stroke_style,stroke_width,contour_width

    # font_args = [stroke_style,stroke_width]
    font_args = [stroke_style,cairo.FONT_WEIGHT_NORMAL]
    # surface = cairo.ImageSurface.create_from_png(buffer)
    # surface = cairo.ImageSurface.create_from_png(buffer)

    # title="this is a \n test!"

    (x_bearing, y_bearing, text_w, text_h,
     x_advance, y_advance) = text_extent(font, font_size, line, *font_args)
    # surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(text_w+6), int(text_h+6))
    surface = cairo.SVGSurface("example.svg", int(text_w+6), int(text_h+6))
    ctx = cairo.Context(surface)
    ctx.select_font_face(font, *font_args)
    ctx.set_font_size(font_size)

    # print "x_bearing:{}, y_bearing:{}, text_w:{}, text_h:{},x_advance:{}, y_advance:{}".format(x_bearing, y_bearing, text_w, text_h,
    #  x_advance, y_advance)
    print text_w,text_h

    ctx.select_font_face(font, *font_args)
    ctx.set_source_rgb(0.0, 0.0, 0.0)
    ctx.move_to(3, text_h + 3)
    ctx.set_font_size(font_size)
    ctx.text_path(line)
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

    # ctx.move_to(-x_bearing,text_h + y_bearing)
    # ctx.stroke()
    ctx.fill()#_preserve()
    font_args = [stroke_style, cairo.FONT_WEIGHT_BOLD]
    # (x_bearing, y_bearing, text_w, text_h,
    #  x_advance, y_advance) = text_extent(font, font_size, line, *font_args)
    ctx.select_font_face(font, *font_args)
    ctx.set_source_rgb(stroke_color[0], stroke_color[1], stroke_color[2])
    ctx.move_to(3, text_h + 3)
    ctx.text_path(line)
    ctx.set_font_size(font_size)
    # ctx.move_to(-x_bearing, -y_bearing)
    # ctx.stroke()
    ctx.fill()

    # ctx.stroke()
    # ctx.set_line_width(contour_width)
    #

    surface.write_to_png(path + '/{}_{}.bmp'.format(lang,i))
    with open(path + '/gt_{}_{}.txt'.format(lang,i),'w') as f:
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
    # for lang in ['en','ja','es','fr','de','it']:
    for lang in ['ja','es']:
        print "processing language {}".format(lang)
        path_file = glob.glob('../data/{}_video_caption_*'.format(lang))
        print path_file
        with open(path_file[0],'rb') as f:
            data_paths = pickle.load(f)
        print len(data_paths)
        cnt = 0
        for [video_path, caption_path] in data_paths:
            filename = video_path.split('/')[-1].split('.')[0]
            output_path = '../data/captions/{}/'.format(lang) + filename + '/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            elif len(os.listdir(output_path) ) != 0:
                cnt += 1
                print "this file has already been processed!"
                continue


            print "processing {}th file, name:{}...".format(cnt,filename)
            r_c = requests.get(caption_path, allow_redirects=True)
            caption_content = r_c.content
            # print caption_content
            encode_format = chardet.detect(caption_content)['encoding']
            print "encode_format:",encode_format
            # caption_content = caption_content.decode(encode_format)  # .decode('utf-16')#.decode('latin-1')
            # print caption_content
            try:
                frame_text,lang_det = convert_smi_to_txt(caption_content,output_path)
            except:
                print sys.exc_info()[0], ":", sys.exc_info()[1]
                print "error accurs! jump to next file"
                with open("error_file_log_{}.txt".format(lang), 'a') as f:
                    f.write(video_path + '\n')
                f.close()
                cnt += 1
                continue
            if lang !=lang_det:

                print "language type not agree, need to check!!!!"
                with open('languge_disagree.txt','w+') as f:
                    f.write(output_path)
                f.close()

                        # generate_caption_png(line,lang_det,cnt,output_path)


            # add_caption_to_frame(video_path, caption_content, output_path)
            cnt +=1
            if cnt == 1000:
                break
        print "done!"
#
#
# if __name__ == '__main__':
#     generate_caption_png("hello world!", 'en', 0, '.')
