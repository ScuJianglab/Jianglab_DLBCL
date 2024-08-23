from os import mkdir
import os

BASEDIR = os.path.dirname(__file__)

seg_parameters = {'seg_params': {'seg_level': -1,
                                 'sthresh': 12,
                                 'mthresh': 1,
                                 'close': 8,
                                 'use_otsu': False,
                                 'keep_ids': 'none',
                                 'exclude_ids': 'none'},
                  'filter_params': {'a_t': 30, 'a_h': 16, 'max_n_holes': 0},
                  'patch_params': {'use_padding': True, 'contour_fn': 'multi_pt'},
                  'vis_params': {'vis_level': -1, 'line_thickness': 125}}


class Config(object):
    def __init__(self):
        # 'patches/14S37393-001_14336_28672.jpg'
        # WSI_files'
        # self.source = '/home1/alfred/dataset/wsi/zhongshan_tissue_lin_2310/colornorm'
        self.source = '/homellm8t/zhaoxz/IHC_one_slide'  # '/public/home/douke/CN/SOC-1'
        self.target_source = '/homellm8t/zhaoxz/HE_merge'  # '/public/home/douke/CN/SOC-1'
        self.normalizer_path = os.path.join(BASEDIR, 'normalizer.pkl')
        self.target_slide_name = 'Train-001.ndpi'
        self.zoom_level = 'x40'
        self.patch_size = 1024
        self.batch_size_fit = 1
        self.batch_size_transform = 20
        self.percentile = 95  # for luminosity standardize
        self.device = 'cuda'  # 'cuda'
        self.sthresh_fit = 16
        self.sthresh_transform = 8
        self.div = 3
        self.workers = 20

        self.get_thumb = True  # get thumb for every slide for quick look
        if self.get_thumb:
            if not os.path.exists(os.path.join(self.source, 'quick_check')):
                mkdir(os.path.join(self.source, 'quick_check'))
        self.save_dir = os.path.join(self.source, 'normalized_slides')
        if not os.path.exists(os.path.join(self.source, 'normalized_slides')):
            mkdir(os.path.join(self.source, 'normalized_slides'))


if __name__ == '__main__':
    print('basic configurations')
