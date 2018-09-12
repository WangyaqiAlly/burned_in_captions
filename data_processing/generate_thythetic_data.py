import numpy as np
import random
import pandas as pd
import sqlalchemy as sa
# import matplotlib.pyplot as plt
# from IPython.display import HTML, display
import requests
import re
import os
# from add_caption import add_caption_to_frame
import pickle
def get_filename_from_cd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
        return None
    fname = re.findall('filename=(.+)', cd)
    if len(fname) == 0:
        return None
    return fname[0]
pd.set_option("display.max_colwidth",9999)

 # %matplotlib inline
# %matplotlib notebook

def read_sql(self, ssql, nidxcol=0):
    df = pd.read_sql(ssql, self)
    return df.set_index(list(df.columns)[:nidxcol]) if nidxcol else df

sa.engine.Engine.read_sql = read_sql

dcalabash = '/calabash/MezzanineArchive'
httpvideo = 'http://httpcalabash.prod.hulu.com/MezzanineArchive'

# theR = sa.create_engine('mysql://homez-ro:hulu12312@ther-db-slave.dc.prod.hulu.com/hulu?charset=utf8', convert_unicode=True)
theR = sa.create_engine('mysql://homez-ro:hulu12312@ther-db-homez-slave.prod.hulu.com/hulu?charset=utf8', convert_unicode=True)

print "1.start loading..."
videos_withCaptionfile = theR.read_sql('''
    select
        a.id as asset_id, a.series_id, a.season_id, a.episode_number, a.title, a.programming_type,
        v.id as video_id, vc.language, f.type, f.path, f.name,fc.type as caption_type, fc.text_language,
        fc.path as caption_path, fc.name as caption_name
    from asset a
        join video v on a.id = v.asset_id
        join video_caption vc on vc.video_id = v.id
        join file f on v.id = f.video_id
        join file fc on v.id = fc.video_id
    where
        a.is_deleted = 0 and v.is_deleted = 0 and v.is_active = 1 and vc.is_deleted = 0 and f.is_deleted = 0
        and a.programming_type in ('Full Episode','Full Movie') and v.regions = 'US'
        and vc.has_caption_file = 1
        and fc.text_language = 'ja'
        and f.type in ('p005', 'p06', 'p007')
        and f.path is not null and f.name is not null
        and fc.type = 'sami'
        and fc.path is not null and fc.name is not null
    order by a.series_id, a.season_id, a.episode_number, a.id, f.type desc
''')
print "2.grouping..."
# and vc.language = 'en'
# and vc.language = 'ja'
# and fc.text_language = 'ja'
# and fc.text_language = 'fr'
# remove redundant records (caused by multiple video files per asset_id, repeated asset_id, etc.)
videos_withCaptionfile = videos_withCaptionfile.groupby(by='asset_id', as_index=False, sort=False).first()

print "videos_withCaptionfile.shape:" ,videos_withCaptionfile.shape
# display(videos_withCaptionfile)


# Ref: language code table: http://www.lingoes.cn/zh/translator/langcode.htm
# display(videos_withCaptionfile.groupby('text_language').size())
print(videos_withCaptionfile.groupby('text_language').size())

# display(videos_withCaptionfile.groupby('language').size())
print(videos_withCaptionfile.groupby('language').size())
#
#
#
#
# assert 0
video_caption_paths = []
if videos_withCaptionfile.shape[0] > 100:
    random_idx = random.sample(range(videos_withCaptionfile.shape[0]), 100)
else:
    random_idx = range(videos_withCaptionfile.shape[0])
print "3.sample and generate frames"
print random_idx
print '======================================'
print
print
# for _, row in videos_withCaption.iloc[:].iterrows():
for _, row in videos_withCaptionfile.iloc[random_idx].iterrows():
    print 'asset_id: %s, series_id: %s, season_id; %s, episode_number: %s' \
          % (row['asset_id'], row['series_id'], row['season_id'], row['episode_number'])
    print 'title: %s' % (row['title'])
    print 'programming_type: %s, language: %s' % (row['programming_type'], row['language'])

    asset_id = row['asset_id']
    video_path = '%s/%s%s' % (httpvideo, row['path'], row['name'])
    caption_path = '%s/%s%s' % (httpvideo, row['caption_path'], row['caption_name'])
    # caption_path =  '%s/%s' % '(httpvideo,487/4378487/7612996_TCFTV_BurnNotice_BCI179_V2_en.smi')
    # r = requests.get(video_path, allow_redirects=True)
    # r_c = requests.get(caption_path, allow_redirects=True)
    # filename = get_filename_from_cd(r.headers.get('content-disposition'))
    # folder_name = '../data/' + filename[:-4]
    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)
    # # data_path = './data/english/'+filename[:-4]+'/'
    # # if not os.path.exists(data_path):
    # #     os.makedirs(data_path)
    # print "synthetic data will be saved to:", data_path
    video_caption_paths.append([video_path,caption_path])


    # add_caption_to_frame(video_path,r_c.content,data_path)

with open("../data/ja_video_dir_random_{}_new".format(len(video_caption_paths)),'wb') as f:
    pickle.dump(video_caption_paths,f)

    #     open(filename, 'wb').write(r.content)
    #     open(filename_caption, 'wb').write(r_c.content)
    # show_video(asset_id, video_path, start_time=0)
    print "done!"
    print
    print