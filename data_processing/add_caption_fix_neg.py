

from __future__ import unicode_literals

import os, StringIO
import re
import codecs
import subprocess
import glob
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw,ImageFont
import cairo
from smi_extract import Subtitle, Smi
import ctypes as ct
import random
import math
import pickle
import requests
import sys
from sys import stdout
import shutil
import chardet
color_bar= {'yellow':(255,255,0),
            'white':(255,255,255),
            'black':(0,0,0),
            'blue':(0,0,255)
            }

# fonts= ['Arial', 'Calibri', 'Helvetica', 'Tahoma' , 'Verdan',]

######random set  parameters to ensure the diversity of the fonts##########
font_family = ["arial","serif", "sans-serif",  "monospace","times new roman","american typewriter"]
font_family_ja = ["yugothic","yukyokasho","yukyokasho yoko","yumincho","yumincho +36p kana","tsukushi a round gothic","tsukushi a round gothic"]

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



def process_img(img, title,path,font, stroke_color,stroke_style,stroke_width,contour_width,font_size,line_width,location):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img =  Image.fromarray(img)
    # draw = ImageDraw.Draw( img )
    w, h = img.size

    # font = "american typewriter"
    # font_size = 20.0
    font_args = [stroke_style,stroke_width]

    # surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(text_width), int(text_height))
    buffer = StringIO.StringIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)


    surface = cairo.ImageSurface.create_from_png(buffer)

    ctx = cairo.Context(surface)
    ctx.select_font_face(font, *font_args)
    # ctx.set_font_size(font_size)
    # ctx.move_to(-x_bearing, -y_bearing)
    # ctx.set_font_face(font_face)
    ctx.set_font_size(font_size)
    # title="this is a \n test!"
    lines = title.split('\n')
    n_line = len(lines)
    boxes = []
    for i, line in enumerate(lines):
        (x_bearing, y_bearing, text_w, text_h,
         x_advance, y_advance) = text_extent(font, font_size, line, *font_args)
        # print "x_bearing:{}, y_bearing:{}, text_w:{}, text_h:{},x_advance:{}, y_advance:{}".format(x_bearing, y_bearing, text_w, text_h,
        #  x_advance, y_advance)


        if location == 'bt':
            ctx.move_to((w - text_w) // 2, h - (n_line-i) * text_h*line_width)
            boxes.append([(w - text_w) // 2, h - (n_line-i) * text_h*line_width,text_w, text_h])
        elif location == 'tp':
            ctx.move_to((w - text_w) // 2, i * text_h * line_width)
            boxes.append([(w - text_w) // 2, i * text_h * line_width,text_w, text_h])
        else:
            raise RuntimeError("undefined location, currently only support 'bt' and 'tp'.")
        # ctx.text_path(title)
        ctx.set_source_rgb(stroke_color[0], stroke_color[1], stroke_color[2])
        ctx.text_path(line)
        print "burning line:{}".format(line)
        ctx.fill_preserve()
        ctx.set_source_rgb(0.0, 0.0, 0.0)
        ctx.set_line_width(contour_width)
        ctx.stroke()
    if len(boxes)>0:
        with open(path + '.pkl', 'wb') as f_bb:
            pickle.dump(boxes, f_bb)
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

    surface.write_to_png(path+'.png')

    # font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 15)
    # text_w, text_h = draw.textsize(title, font)
    # draw.text(((w - text_w) // 2, h - text_h), title, (255,255,255), font=font)
    # # draw.text((REQ_WIDTH, REQ_HEIGHT), title, (255,255,255), font=font)
    #
    # img.save(path)

    return img

def neg_sample_generate(cap,starts, ends,total_frames,output_folder):
    '''
    generate negtive frames from where no caption appears,
    which means the gap between the timestamp of two adjacent captions
    '''
    gap = []
    for i in xrange(len(starts)-1):
        if starts[i+1]-ends[i]>1:
            gap +=range(ends[i]+1,starts[i+1])
    print len(gap)
    if len(gap)>= 100:
        neg_frames_idx =random.sample(gap,100)
    else:
        print "no gap or gap is  not long enough,get neg samples from begining and ends!!!!!!"
        neg_frames_idx = gap+random.sample(range(0,starts[0])+range(ends[-1],int(total_frames)),100 - len(gap))
    print neg_frames_idx
    for idx in neg_frames_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret_nocap, frame_nocap = cap.read()
        cv2.imwrite(output_folder + "/neg_frame_{}.png".format(idx), frame_nocap)






def add_caption_to_frame(lang,video_path,caption_content,output_folder):
    #####load smi and video

    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print "fps:{}, total_frames:{}".format(fps, total_frames)

    smi = Smi(caption_content)
    frame_text = smi.convert('frame', lang='ENCC')
    if len(frame_text) == 0:
        frame_text = smi.convert('frame', lang='ENUSCC')
    if len(frame_text) == 0:
        frame_text = smi.convert('frame', lang='SUBTTL')
    if len(frame_text) == 0:
        frame_text = smi.convert('frame', lang='JAJPCC')
    print "totoal captions:", len(frame_text)  # ,frame_text[:]
    if len(frame_text)<100:
        print "warning, subtitiles is not enough for generating!"
        with open("error_file_log.txt",'a') as f:
            f.write(video_path+'\n')
        f.close()
        return

    #convert timestamp(msec) to frame number
    for caption in frame_text:
        caption[0] = int(float(caption[0]) / 1000.0 * fps)
        caption[1] = int(float(caption[1]) / 1000.0 * fps)



    assert len(frame_text) >= 100, "warning, subtitiles is not enough for generating!"
    print "totoal captions:", len(frame_text)  # ,frame_text[:]
    overflow_lines = 0
    while frame_text[-1][1] > total_frames:
        print "warning !caption overflow!!! {}th line!".format(overflow_lines)
        frame_text = frame_text[:-1]
        overflow_lines += 1
        assert overflow_lines <= 10, "caption overflow too much(more than 10 lines)!! "
    assert len(frame_text) >= 100, "warning, subtitiles is not enough for generating!"
    print "totoal captions:", len(frame_text)  # ,frame_text[:]

    starts = [x[0] for x in frame_text]
    ends = [x[1] for x in frame_text]
    assert ends[ -1] <= total_frames, "caption overflow!!! caption frames end at:{}, while {} frames in this video".format(
        ends[-1], total_frames)

    # #####set the 'caption style' parameters for each video####
    # # font_path = random.choice(font_paths)
    # if lang == 'ja':
    #     font = random.choice(font_family_ja)
    # else:
    #     font = random.choice(font_family)
    #
    #
    # stroke_color = random_pick([(1.0,1.0,1.0),(1.0,1.0,0)],[0.9,.1])
    # stroke_style = random_pick([cairo.FONT_SLANT_NORMAL,cairo.FONT_SLANT_ITALIC,cairo.FONT_SLANT_OBLIQUE],[0.6,0.2,0.2])
    # stroke_width = cairo.FONT_WEIGHT_BOLD
    # contour_width  =  math.fabs(random.gauss(0.5, 0.2))
    #
    # font_size = random.gauss(25, 5)
    # line_width = 1.0 + math.fabs(random.gauss(0, 0.2))
    # location = random_pick(['bt','tp'],[1.0,0])
    # # font_face = create_cairo_font_face_for_file(font_path, 0)
    #
    # print "'caption style' in this video:\n font:{},\nstroke_color:{},stroke_style:{},stroke_width:{}, \ncontour_width:{},\n," \
    #       " font_size:{}, \n line_width:{}, \n location:{} ".format(font,stroke_color,stroke_style,stroke_width, contour_width, font_size,line_width,location)
    #
    # print "pos samples generating..."
    # random_idx = random.sample(range(len(frame_text)), 100)
    # random_idx.sort()
    # cnt = 0
    # for i in xrange(len(random_idx)):
    #     stdout.write("\rprocessing %d th frame..."% i)
    #     stdout.flush()
    # # for caption in frame_text[random_idx]:
    #     caption = frame_text[random_idx[i]]
    #     # next_caption = frame_text[i+1]
    #     # #####set the 'caption style' parameters for each frame####
    #
    #     # font_path = random.choice(font_paths)
    #     # stroke_color = random_pick([(1.0, 1.0, 1.0), (1.0, 1.0, 0)], [0.9, .1])
    #     # stroke_style = random_pick([cairo.FONT_SLANT_NORMAL, cairo.FONT_SLANT_ITALIC, cairo.FONT_SLANT_OBLIQUE],
    #     #                            [0.1, 0.1, 0.8])
    #     # contour_width = math.fabs(random.gauss(0.5, 0.2))
    #     #
    #     # font_size = random.gauss(25, 5)
    #     # line_width = 1.0 + math.fabs(random.gauss(0, 0.2))
    #     # location = random_pick(['bt', 'tp'], [0.99, .01])
    #     # # font_face = create_cairo_font_face_for_file(font_path, 0)
    #     #
    #     # print "'caption style' in this video:\n font:{},\nstroke_color:{}, \ncontour_width:{},\n" \
    #     #       " font_size:{}, \n line_width:{}, \n location:{} ".format(font_path, stroke_color, contour_width,
    #     #                                                                 font_size, line_width, location)
    #
    #     # print "current caption:",caption
    #     # # debug
    #     # # debug
    #     # if cnt == 10:
    #     #     assert 0
    #     start_frame_idx= caption[0]
    #     end_frame_idx=  caption[1]
    #     duration = end_frame_idx - start_frame_idx
    #     sampled_idx = int((start_frame_idx+end_frame_idx)/2)
    #
    #     # print "duration:", duration
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, sampled_idx )
    #     ret, frame = cap.read()
    #     print "frame {} , sbutitle:{} ".format(sampled_idx, caption[2])
    #     process_img(frame,caption[2],output_folder+"/frame_{}".format(sampled_idx),font,stroke_color,stroke_style,stroke_width,contour_width,font_size,line_width,location)
    #     # next_msec = next_caption[0]
    #     # print "test the frame order!!! end_msec:{} ,next_msec:{}".format(end_msec, next_msec)
    #     # no_caption_msec = int((end_msec + next_msec) / 2)
    #
    #     # cv2.putText(frame, caption[2],bottomLeftOrigin=True)
    #     # cv2.imwrite(save_path+"/test_{}.jpg".format(start_frame), frame)
    #     cnt +=1
    #     # Set waitKey
    #     # cv2.waitKey()
    #     # plt.(frame, cmap='gray')
    #     # plt.show()
    #
    # # cv2.imwrite("path_where_to_save_image", frame)
    # #
    # # cnt = 0
    # # total_frames = 0
    # # while(cap.isOpened() and cnt < end_f):
    # #     ret, frame = cap.read()
    # #     if cnt > start_f and cnt % (int(fps)*10) == 0:
    # #         print 'processing {}th frame'.format(cnt)
    # #         cv2.imwrite(os.path.join(save_path,str(cnt)+'.jpg'),frame)
    # #         total_frames +=1
    # #     cnt += 1
    # # print "done! total frames: {}".format(total_frames)
    print "neg samples generating..."
    neg_sample_generate(cap, starts, ends, total_frames, output_folder)

    cap.release()
    # cv2.destroyAllWindows()



if __name__ == '__main__':
    lang = "ja"
    with open('../data/ja_video_dir_random_100','rb') as f:
        data_paths = pickle.load(f)
    print len(data_paths)
    cnt = 0

    for [video_path, caption_path] in data_paths:
        filename = video_path.split('/')[-1].split('.')[0]
        output_path = '../data/pngdataset/images/'+lang+'/' + filename + '/'
        if filename  not in ["agave3024126_24794627_H264_1000"]:
            cnt+=1
            continue
        # if os.path.exists(output_path):
        #     if len(glob.glob(output_path+'/*neg'))==300:
        #         cnt += 1
        #         print "this video has already been processed!"
        #         continue
        #     else:
        #         shutil.rmtree(output_path)
        #         os.makedirs(output_path)
        else:
            if os.path.exists(output_path):
                negfiles = glob.glob(output_path+'/neg*')
                for negfile in negfiles:
                    os.remove(negfile)
                    print "delete file", negfile
                # shutil.rmtree(output_path)
            # os.makedirs(output_path)

        print "processing {}th video, name:{}...".format(cnt,filename)

        r_c = requests.get(caption_path, allow_redirects=True)
        caption_content = r_c.content

        if len(caption_content) ==0:
            print "empty smi! this file should be deleted !"
            with open("error_file_log_{}_empty.txt".format(lang), 'a') as f:
                f.write(video_path + '\n')
            f.close()
            continue
        encode_format = chardet.detect(caption_content)['encoding']
        if encode_format is None:
            encode_format = 'utf-8'
        print encode_format
        # caption_content = caption_content.decode('latin-1')

        # print caption_content
        # with open(output_path + '/caption.txt', 'w') as f:
        #     f.write(caption_content)
        # f.close()



        # print caption_content

        try:
            add_caption_to_frame(lang,video_path, caption_content, output_path)
        except:
            print sys.exc_info()[0],":",sys.exc_info()[1]
            print "error accurs! jump to next file"
            with open("error_file_log_{}.txt".format(lang), 'a') as f:
                f.write(video_path + '\n')
            f.close()
            pass
        cnt +=1
    print "done!"

