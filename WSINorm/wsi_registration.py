from utils.registration import get_img_in_source, RegistrationQuadTree, transform_min_corr
from pathlib import Path
import numpy as np
import openslide
import logging
import os
from CLAM.create_patches_fp import seg_and_patch
import h5py
from PIL import Image
import cv2
import multiprocessing
from multiprocessing import Pool, Lock, Manager
from functools import partial
import tqdm
# 设置更高的像素限制
logging.basicConfig(level='INFO')

data_dir = '/homellm8t/zhaoxz/'

other_modal = '/homellm8t/zhaoxz/HE_merge_small'
patch_size = 1024
fmt = 'tiff'
num_processes = 16

source = os.path.join(data_dir, 'IHE_small_regis')
target = os.path.join(data_dir, other_modal)

save_dir = os.path.join(data_dir, 'samll_HE/masked_patch')
directories = {'source': source,
               'save_dir': save_dir,
               'patch_save_dir': os.path.join(save_dir, 'patches'),
               'mask_save_dir': os.path.join(save_dir, 'masks'),
               'stitch_save_dir': os.path.join(save_dir, 'stitches')}

patch_he_img_dir = os.path.join(directories['patch_save_dir'], 'IHC')
patch_other_img_dir = os.path.join(directories['patch_save_dir'], other_modal)

reg_parameters = {
    # feature extractor parameters
    "point_extractor": "sift",  # orb , sift
    "maxFeatures": 1024,
    "crossCheck": False,
    "flann": False,
    "ratio": 0.7,
    "use_gray": True,

    # QTree parameter
    "homography": True,
    "filter_outliner": False,
    "debug": True,
    "target_depth": 1,
    "run_async": True,
    "thumbnail_size": (2048, 2048)
}
# reg_parameters = {
#                 # feature extractor parameters
#                 "point_extractor": "sift",  #orb , sift
#                 "maxFeatures": 4096, 
#                 "crossCheck": False, 
#                 "flann": False,
#                 "ratio": 0.7, 
#                 "use_gray": False,

#                 # QTree parameter 
#                 "homography": True,
#                 "filter_outliner": False,
#                 "debug": True,
#                 "target_depth": 0,
#                 "run_async": True,
#                 "thumbnail_size": (1024, 1024)
#             }


seg_parameters = {'seg_params': {'seg_level': -1,
                                 'sthresh': 12,
                                 'mthresh': 1,
                                 'close': 8,
                                 'use_otsu': False,
                                 'keep_ids': 'none',
                                 'exclude_ids': 'none'},
                  'filter_params': {'a_t': 30, 'a_h': 16, 'max_n_holes': 8},
                  'patch_params': {'use_padding': True, 'contour_fn': 'four_pt'},
                  'vis_params': {'vis_level': -1, 'line_thickness': 125}}

for key, val in directories.items():
    print("{} : {}".format(key, val))
    if key not in ['source']:
        os.makedirs(val, exist_ok=True)
if not os.path.exists(patch_he_img_dir):
    os.mkdir(patch_he_img_dir)
if not os.path.exists(patch_other_img_dir):
    os.mkdir(patch_other_img_dir)

seg_parameters['seg_params']['sthresh'] = 0
seg_times, patch_times = seg_and_patch(**directories, **seg_parameters,
                                       patch_size=patch_size, step_size=patch_size,
                                       seg=True, use_default_params=False, save_mask=True,
                                       stitch=True, patch_level=0, patch=True,
                                       process_list=None, auto_skip=True)
exist_he_img_list = [i.replace('.png', '') for i in os.listdir(patch_he_img_dir)]
exits_other_img_list = [i.replace('.png', '') for i in os.listdir(patch_other_img_dir)]


def process_patch(loc, name, patch_he_img_dir, patch_other_img_dir, exist_he_img_list, exits_other_img_list,
                  he_path, other_path, qtree, patch_size, lock):
    patch_name = name + '_' + '_'.join(loc.astype(str))
    patch_he_img_path = os.path.join(patch_he_img_dir, patch_name + '.png')
    patch_other_img_path = os.path.join(patch_other_img_dir, patch_name + '.png')
    if (patch_name not in exist_he_img_list) or (patch_name not in exits_other_img_list):
        he_slide = openslide.open_slide(he_path)
        other_slide = openslide.open_slide(other_path)

        try:
            he_img = he_slide.read_region(location=loc, size=(patch_size, patch_size), level=0)
            he_img = np.array(he_img)
            other_img = get_img_in_source(x=loc[0], y=loc[1], w=patch_size, h=patch_size, target=other_slide,
                                          qtree=qtree)
        except (openslide.lowlevel.OpenSlideError, NotImplementedError) as e:
            print('Error: %s, %s, info: %s' % (name, loc, e))
            he_slide.close()
            other_slide.close()
            del he_slide
            del other_slide
            return
        he_slide.close()
        other_slide.close()
        del he_slide
        del other_slide

        r = np.abs(np.corrcoef(cv2.cvtColor(he_img, cv2.COLOR_BGR2GRAY).flatten(),
                               cv2.cvtColor(other_img, cv2.COLOR_BGR2GRAY).flatten())[0, 1])
        tf_m, tf_other_img = transform_min_corr(other_img, he_img)  # transform other_img
        tf_r = np.abs(np.corrcoef(cv2.cvtColor(he_img, cv2.COLOR_BGR2GRAY).flatten(),
                                  cv2.cvtColor(tf_other_img, cv2.COLOR_BGR2GRAY).flatten())[0, 1])
        if (tf_r > (r + 0.01)) and (1.15 > np.linalg.det(tf_m[:2, :2]) > 0.85):
            print('Abs Corr optimized from %.2f to %.2f' % (r, tf_r))
            other_img = tf_other_img

        he_img = Image.fromarray(he_img)
        other_img = Image.fromarray(other_img)
        if (patch_name not in exist_he_img_list):
            he_img.convert('RGB').save(patch_he_img_path, quality=90)
        if (patch_name not in exits_other_img_list):
            other_img.convert('RGB').save(patch_other_img_path, quality=90)
            lock.acquire()
            with open(os.path.join(directories['patch_save_dir'], 'info.txt'), 'a') as f:
                f.write(f'{patch_name}, {other_modal}, {r: .2f}, {tf_r: .2f}\r\n')
            lock.release()


for h5f in os.listdir(directories['patch_save_dir']):
    if not '.h5' in h5f:
        continue
    f = h5py.File(os.path.join(directories['patch_save_dir'], h5f), 'r')
    coords = f['coords']
    level_dim = coords.attrs['level_dim']
    name = coords.attrs['name']
    patch_size = coords.attrs['patch_size']
    level = coords.attrs['patch_level']
    patch_coords = coords[()]
    f.close()

    he_path = os.path.join(source, name + '.' + fmt)
    other_path = os.path.join(target, name + '.' + fmt)

    if (not os.path.exists(he_path)) or (not os.path.exists(other_path)):
        continue
    logging.info('Patching %s' % h5f)

    # TODO: skip slides processed
   
    qtree = RegistrationQuadTree(source_slide_path=Path(he_path),
                                 target_slide_path=Path(other_path), **reg_parameters)

    pool = Pool(processes=num_processes)
    lock = Manager().Lock()  # 
    num_processes=15
    process_patch_partial = partial(process_patch, name=name, patch_he_img_dir=patch_he_img_dir,
                                    patch_other_img_dir=patch_other_img_dir, exist_he_img_list=exist_he_img_list,
                                    exits_other_img_list=exits_other_img_list,
                                    he_path=he_path, other_path=other_path,
                                    qtree=qtree, patch_size=patch_size, lock=lock)

    for _ in tqdm.tqdm(pool.map(process_patch_partial, patch_coords), total=len(patch_coords)):
        pass
    pool.close()
    pool.join()
