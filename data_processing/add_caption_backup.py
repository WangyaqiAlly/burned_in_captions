

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
import cairo
import random
import math

color_bar= {'yellow':(255,255,0),
            'white':(255,255,255),
            'black':(0,0,0),
            'blue':(0,0,255)
            }

# fonts= ['Arial', 'Calibri', 'Helvetica', 'Tahoma' , 'Verdan',]

######random set  parameters to ensure the diversity of the fonts##########
font_paths= glob.glob("/usr/share/fonts/truetype/*/*.ttf")
'''
/usr/share/fonts/truetype/freefont/FreeMonoBoldOblique.ttf
/usr/share/fonts/truetype/freefont/FreeMono.ttf
/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf
/usr/share/fonts/truetype/freefont/FreeSerifBoldItalic.ttf
/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf
/usr/share/fonts/truetype/freefont/FreeSansBold.ttf
/usr/share/fonts/truetype/freefont/FreeMonoOblique.ttf
/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf
/usr/share/fonts/truetype/freefont/FreeSans.ttf
/usr/share/fonts/truetype/freefont/FreeSansBoldOblique.ttf
/usr/share/fonts/truetype/freefont/FreeSerif.ttf
/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf
/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf
/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf
/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf
/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf
'''


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability: break
    return item


class PycairoContext(ct.Structure):
    _fields_ = \
        [
            ("PyObject_HEAD", ct.c_byte * object.__basicsize__),
            ("ctx", ct.c_void_p),
            ("base", ct.c_void_p),
        ]

_initialized = False
def create_cairo_font_face_for_file (filename, faceindex=0, loadoptions=0):
    "given the name of a font file, and optional faceindex to pass to FT_New_Face" \
    " and loadoptions to pass to cairo_ft_font_face_create_for_ft_face, creates" \
    " a cairo.FontFace object that may be used to render text with that font."
    global _initialized
    global _freetype_so
    global _cairo_so
    global _ft_lib
    global _ft_destroy_key
    global _surface

    CAIRO_STATUS_SUCCESS = 0
    FT_Err_Ok = 0

    if not _initialized:
        print "not initialized!!!"
        # find shared objects
        _freetype_so = ct.CDLL("libfreetype.so.6")
        _cairo_so = ct.CDLL("libcairo.so.2")
        _cairo_so.cairo_ft_font_face_create_for_ft_face.restype = ct.c_void_p
        _cairo_so.cairo_ft_font_face_create_for_ft_face.argtypes = [ ct.c_void_p, ct.c_int ]
        _cairo_so.cairo_font_face_get_user_data.restype = ct.c_void_p
        _cairo_so.cairo_font_face_get_user_data.argtypes = (ct.c_void_p, ct.c_void_p)
        _cairo_so.cairo_font_face_set_user_data.argtypes = (ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p)
        _cairo_so.cairo_set_font_face.argtypes = [ ct.c_void_p, ct.c_void_p ]
        _cairo_so.cairo_font_face_status.argtypes = [ ct.c_void_p ]
        _cairo_so.cairo_font_face_destroy.argtypes = (ct.c_void_p,)
        _cairo_so.cairo_status.argtypes = [ ct.c_void_p ]
        # initialize freetype
        _ft_lib = ct.c_void_p()
        status = _freetype_so.FT_Init_FreeType(ct.byref(_ft_lib))
        if  status != FT_Err_Ok :
            raise RuntimeError("Error %d initializing FreeType library." % status)
        #end if


        #end PycairoContext

        _surface = cairo.ImageSurface(cairo.FORMAT_A8, 0, 0)
        _ft_destroy_key = ct.c_int() # dummy address
        _initialized = True
    #end if

    ft_face = ct.c_void_p()
    cr_face = None
    try :
        # load FreeType face
        status = _freetype_so.FT_New_Face(_ft_lib, filename.encode("utf-8"), faceindex, ct.byref(ft_face))
        if status != FT_Err_Ok :
            raise RuntimeError("Error %d creating FreeType font face for %s" % (status, filename))
        #end if

        # create Cairo font face for freetype face
        cr_face = _cairo_so.cairo_ft_font_face_create_for_ft_face(ft_face, loadoptions)
        status = _cairo_so.cairo_font_face_status(cr_face)
        if status != CAIRO_STATUS_SUCCESS :
            raise RuntimeError("Error %d creating cairo font face for %s" % (status, filename))
        #end if
        # Problem: Cairo doesn't know to call FT_Done_Face when its font_face object is
        # destroyed, so we have to do that for it, by attaching a cleanup callback to
        # the font_face. This only needs to be done once for each font face, while
        # cairo_ft_font_face_create_for_ft_face will return the same font_face if called
        # twice with the same FT Face.
        # The following check for whether the cleanup has been attached or not is
        # actually unnecessary in our situation, because each call to FT_New_Face
        # will return a new FT Face, but we include it here to show how to handle the
        # general case.
        if _cairo_so.cairo_font_face_get_user_data(cr_face, ct.byref(_ft_destroy_key)) == None :
            status = _cairo_so.cairo_font_face_set_user_data \
              (
                cr_face,
                ct.byref(_ft_destroy_key),
                ft_face,
                _freetype_so.FT_Done_Face
              )
            if status != CAIRO_STATUS_SUCCESS :
                raise RuntimeError("Error %d doing user_data dance for %s" % (status, filename))
            #end if
            ft_face = None # Cairo has stolen my reference
        #end if

        # set Cairo font face into Cairo context
        cairo_ctx = cairo.Context(_surface)
        cairo_t = PycairoContext.from_address(id(cairo_ctx)).ctx
        _cairo_so.cairo_set_font_face(cairo_t, cr_face)
        status = _cairo_so.cairo_font_face_status(cairo_t)
        if status != CAIRO_STATUS_SUCCESS :
            raise RuntimeError("Error %d creating cairo font face for %s" % (status, filename))
        #end if

    finally :
        _cairo_so.cairo_font_face_destroy(cr_face)
        _freetype_so.FT_Done_Face(ft_face)
    #end try

    # get back Cairo font face as a Python object
    face = cairo_ctx.get_font_face()
    return face
#end create_cairo_font_face_for_file


def text_extent(font_face, font_size, text, *args, **kwargs):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 0, 0)
    ctx = cairo.Context(surface)
    # ctx.select_font_face(font, *args, **kwargs)

    ctx.set_font_face(font_face)
    ctx.set_font_size(font_size)
    return ctx.text_extents(text)


def process_img(img, title,path, stroke_color,contour_width,font_size,line_width,location,font_face):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img =  Image.fromarray(img)
    # draw = ImageDraw.Draw( img )
    w, h = img.size

    # font = "Purisa"
    # font_size = 20.0
    font_args = [cairo.FONT_SLANT_NORMAL]

    # surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(text_width), int(text_height))
    buffer = StringIO.StringIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)


    surface = cairo.ImageSurface.create_from_png(buffer)

    ctx = cairo.Context(surface)
    # ctx.select_font_face(font, *font_args)
    # ctx.set_font_size(font_size)
    # ctx.move_to(-x_bearing, -y_bearing)
    ctx.set_font_face(font_face)
    ctx.set_font_size(font_size)

    # title="this is a \n test!"
    lines = title.split('\n')
    n_line = len(lines)

    for i, line in enumerate(lines):
        (x_bearing, y_bearing, text_w, text_h,
         x_advance, y_advance) = text_extent(font_face, font_size, line, *font_args)
        if location == 'bt':
            ctx.move_to((w - text_w) // 2, h - (n_line-i) * text_h*line_width)
        elif location == 'tp':
            ctx.move_to((w - text_w) // 2, i * text_h * line_width)
        else:
            raise RuntimeError("undefined location, currently only support 'bt' and 'tp'.")
        # ctx.text_path(title)
        ctx.set_source_rgb(stroke_color[0], stroke_color[1], stroke_color[2])
        ctx.text_path(line)
        ctx.fill_preserve()
        ctx.set_source_rgb(0.0, 0.0, 0.0)
        ctx.set_line_width(contour_width)
        ctx.stroke()

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

    surface.write_to_png(path)

    # font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 15)
    # text_w, text_h = draw.textsize(title, font)
    # draw.text(((w - text_w) // 2, h - text_h), title, (255,255,255), font=font)
    # # draw.text((REQ_WIDTH, REQ_HEIGHT), title, (255,255,255), font=font)
    #
    # img.save(path)

    return img



#####load smi and video
smi = Smi('mi.smi')
frame_text = smi.convert('frame', lang='ENCC')
print len(frame_text),frame_text[:]


##### add text to frames #####
filenames = glob.glob('ja.mp4')

save_path = 'frames'
if not os.path.exists(save_path):
    os.makedirs(save_path)

cap = cv2.VideoCapture(filenames[0])
fps = int(cap.get(cv2.CAP_PROP_FPS))
fps_f = cap.get(cv2.CAP_PROP_FPS)
total_len =  cap.get(cv2.CAP_PROP_FRAME_COUNT)
print fps,total_len, total_len/fps


#####set the 'caption style' parameters for each video####
font_path = random.choice(font_paths)
stroke_color = random_pick([(1.0,1.0,1.0),(1.0,1.0,0)],[0.9,.1])
contour_width  =  math.fabs(random.gauss(0.5, 0.2))
font_size = random.gauss(22, 5)
line_width = random.gauss(1.0, 0.2)
location = random_pick(['bt','tp'],[0.99,.01])
font_face = create_cairo_font_face_for_file(font_path, 0)
print "'caption style' in this video:\n font:{},\nstroke_color:{}, \ncontour_width:{},\n" \
      " font_size:{}, \n line_width:{}, \n location:{} ".format(font_path,stroke_color, contour_width, font_size,line_width,location)




cnt = 0
for caption in frame_text:
    #####set the 'caption style' parameters for each video####
    font_path = random.choice(font_paths)
    stroke_color = random_pick([(1.0, 1.0, 1.0), (1.0, 1.0, 0)], [0.9, .1])
    contour_width = math.fabs(random.gauss(1, 0.5))
    font_size = random.gauss(20, 5)
    line_width = random.gauss(1.2, 0.2)
    location = random_pick(['bt', 'tp'], [0.99, .01])
    font_face = create_cairo_font_face_for_file(font_path, 0)
    print "'caption style' in this video:\n font:{},\nstroke_color:{}, \nhave_counter:{},\n" \
          " font_size:{}, \n line_width:{}, \n location:{} ".format(font_path, stroke_color, contour_width, font_size,
                                                                    line_width, location)

    # debug
    if cnt == 10:
        assert 0
    start_frame = int(fps_f * caption[0] / 1000.0)
    end_frame = int(fps_f * caption[1] / 1000.0)
    duration = end_frame-start_frame
    print "duration:", duration
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame+end_frame/2))
    ret, frame = cap.read()
    process_img(frame,caption[2],save_path+"/test_{}_sb.png".format(start_frame),stroke_color,contour_width,font_size,line_width,location,font_face)
    # cv2.putText(frame, caption[2],bottomLeftOrigin=True)
    # cv2.imwrite(save_path+"/test_{}.jpg".format(start_frame), frame)
    cnt +=1
    # Set waitKey
    # cv2.waitKey()
    # plt.(frame, cmap='gray')
    # plt.show()

# cv2.imwrite("path_where_to_save_image", frame)
#
# cnt = 0
# total_frames = 0
# while(cap.isOpened() and cnt < end_f):
#     ret, frame = cap.read()
#     if cnt > start_f and cnt % (int(fps)*10) == 0:
#         print 'processing {}th frame'.format(cnt)
#         cv2.imwrite(os.path.join(save_path,str(cnt)+'.jpg'),frame)
#         total_frames +=1
#     cnt += 1
# print "done! total frames: {}".format(total_frames)
cap.release()
# cv2.destroyAllWindows()



