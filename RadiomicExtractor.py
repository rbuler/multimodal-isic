import cv2
import time
import logging
import multiprocessing
import SimpleITK as sitk
from tqdm import tqdm #  , trange
from multiprocessing import Pool
from radiomics import featureextractor
logger = logging.getLogger(__name__)


class RadiomicsExtractor():

    def  __init__(self, param_file: str):
        self.extractor = featureextractor.RadiomicsFeatureExtractor(param_file)

    def get_enabled_image_types(self):
        return list(self.extractor.enabledImagetypes.keys())
    
    def get_enabled_features(self):
        return list(self.extractor.enabledFeatures.keys())

    def extract_radiomics(self, list_of_dicts):

        label = self.extractor.settings.get('label', None)
        img_path = list_of_dicts['image_path']
        seg_path = list_of_dicts['segmentation_path']

        im = cv2.imread(img_path, cv2.IMREAD_COLOR)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = sitk.GetImageFromArray(im_gray)

        sg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        if im.shape[:2] != sg.shape[:2]:
            sg = cv2.resize(sg, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        sg = sitk.GetImageFromArray(sg)
        
        features = self.extractor.execute(im_gray, sg, label=label)
        features_gray = features

        im_red = sitk.GetImageFromArray(im[:, :, 2])
        features_red = self.extractor.execute(im_red, sg, label=label)

        im_green = sitk.GetImageFromArray(im[:, :, 1])
        features_green = self.extractor.execute(im_green, sg, label=label)

        im_blue = sitk.GetImageFromArray(im[:, :, 0])
        features_blue = self.extractor.execute(im_blue, sg, label=label)

        return {
            "grayscale": features_gray,
            "red": features_red,
            "green": features_green,
            "blue": features_blue
        }
    

    def parallell_extraction(self, list_of_dicts: list, n_processes = None):
        logger.info("Extraction mode: parallel")
        if n_processes is None:
            n_processes = multiprocessing.cpu_count() - 1
        start_time = time.time()
        with Pool(n_processes) as pool:
            results = list(tqdm(pool.imap(self.extract_radiomics, list_of_dicts),
                                 total=len(list_of_dicts)))
        end_time = time.time()

        h, m, s = self._convert_time(start_time, end_time)
        logger.info(f" Time taken: {h}h:{m}m:{s}s")

        return results
    

    def serial_extraction(self, list_of_dicts: list):
        logger.info("Extraction mode: serial")
        all_results = []
            # for item in trange(len(train_df)):
        start_time = time.time()
        for item in range(len(list_of_dicts)):
            all_results.append(self.extract_radiomics(list_of_dicts[item]))
        end_time = time.time()

        h, m, s = self._convert_time(start_time, end_time)
        logger.info(f" Time taken: {h}h:{m}m:{s}s")
        return all_results


    def _convert_time(self, start_time, end_time):
        '''
        Converts time in seconds to hours, minutes and seconds.
        '''
        dt = end_time - start_time
        h, m, s = int(dt // 3600), int((dt % 3600 ) // 60), int(dt % 60)
        return h, m, s
