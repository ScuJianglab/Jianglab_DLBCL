import os
import logging
from config import seg_parameters
from CLAM.create_patches_fp import seg_and_patch

logging.basicConfig(level='INFO')


def patching(source, patch_size, sthresh=8):
    save_dir = os.path.join(source, 'masked_patch', 'th_%d' % sthresh)
    directories = {'source': source,
                   'save_dir': save_dir,
                   'patch_save_dir': os.path.join(save_dir, 'patches'),
                   'mask_save_dir': os.path.join(save_dir, 'masks'),
                   'stitch_save_dir': os.path.join(save_dir, 'stitches')}
    seg_mask_dir = os.path.join(directories['mask_save_dir'], 'seg')
    patch_img_dir = os.path.join(directories['patch_save_dir'], 'img')
    patch_label_img_dir = os.path.join(directories['patch_save_dir'], 'label')
    tif_patch_img_dir = os.path.join(directories['patch_save_dir'], 'tif_img')

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)
    if not os.path.exists(seg_mask_dir):
        os.mkdir(seg_mask_dir)
    if not os.path.exists(patch_img_dir):
        os.mkdir(patch_img_dir)
    if not os.path.exists(patch_label_img_dir):
        os.mkdir(patch_label_img_dir)
    if not os.path.exists(tif_patch_img_dir):
        os.mkdir(tif_patch_img_dir)

    seg_parameters['seg_params']['sthresh'] = sthresh
    seg_times, patch_times = seg_and_patch(**directories, **seg_parameters,
                                           patch_size=patch_size, step_size=patch_size,
                                           seg=True, use_default_params=False, save_mask=True,
                                           stitch=True, patch_level=0, patch=True,
                                           process_list=None, auto_skip=True)
