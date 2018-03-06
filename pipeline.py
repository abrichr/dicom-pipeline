"""Pipeline for parsing DICOM and i-contour files"""


# Verbose output
DEBUG = True

# Absolute or relative path to data directory
DATA_DIR = 'final_data'

# Name of csv file containing links between dicoms and contours
LINK_FNAME = 'link.csv'

# Name of column in link file containing DICOM names
DICOM_KEY = 'patient_id'

# Name of column in link file containing contour names
CONTOUR_KEY = 'original_id'

# Alpha transparency of mask overlaid on DICOM
MASK_OVERLAY_ALPHA = 0.25


import logging
log_format = '%(asctime)s : %(name)s : %(levelname)s : %(message)s'
logging.basicConfig(format=log_format, level=logging.WARN)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)


import csv
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
from pprint import pprint, pformat


from parsing import parse_contour_file, parse_dicom_file, poly_to_mask


def _get_cid_by_did(
    link_path=join(DATA_DIR, LINK_FNAME),
    dicom_key=DICOM_KEY,
    contour_key=CONTOUR_KEY
):
  """Read links between dicoms and contours"""

  cid_by_did = {}
  with open(link_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
      dicom_id = row[dicom_key]
      contour_id = row[contour_key]
      cid_by_did[dicom_id] = contour_id
  logger.debug('cid_by_did: %s' % cid_by_did)
  return cid_by_did
  
def get_dicom_mask_tups(
    dicom_dir,
    contour_dir,
    show_masked_dicoms=False
):
  '''
  Function that takes in the directories containing the DICOM and contour files
  and provides an iterator over two-element tuples, where each two-element tuple
  consists of an image and its associated mask as Numpy arrays.
  '''

  cid_by_did = _get_cid_by_did()

  for dicom_id, contour_id in cid_by_did.iteritems():

    # read dicom paths
    dicom_path = join(dicom_dir, dicom_id)
    logger.debug('dicom_path: %s' % dicom_path)
    dicom_path_by_id = {}
    for f in listdir(dicom_path):
      f_path = join(dicom_path, f)
      if not isfile(f_path):
        continue
      try:
        dicom_id = int(f.split('.')[0])
      except:
        logger.warn('Unable to parse dicom_id for file %s' % f)
        continue
      logger.debug('dicom_id: %s, f_path: %s' % (dicom_id, f_path))
      if dicom_id in dicom_path_by_id:
        logger.warn('duplicate dicom_id %s' % dicom_id)
        continue
      dicom_path_by_id[dicom_id] = f_path

    # read contour paths
    contour_path = join(contour_dir, contour_id, 'i-contours')
    contour_path_by_id = {}
    for f in listdir(contour_path):
      f_path = join(contour_path, f)
      if not isfile(f_path):
        continue
      try:
        contour_id = int(f.split('-')[2])
      except:
        logger.warn('Unable to parse mask_id for file %s' % f)
        continue
      if contour_id in contour_path_by_id:
        logger.warn('duplicate contour_id: %s' % contour_id)
        continue
      contour_path_by_id[contour_id] = f_path

    # pair up common ids
    dicom_ids = set(dicom_path_by_id.keys())
    contour_ids = set(contour_path_by_id.keys())
    common_ids = sorted(list(dicom_ids.intersection(contour_ids)))
    logger.debug('common_ids: %s' % pformat(common_ids))
    dicom_contour_path_tups = [
        (dicom_path_by_id[_id], contour_path_by_id[_id]) for _id in common_ids]

    for dicom_path, contour_path in dicom_contour_path_tups:

      # read dicom data
      dcm_dict = parse_dicom_file(dicom_path)
      dicom = dcm_dict['pixel_data']
      width, height = dicom.shape

      # read contour data
      coords_list = parse_contour_file(contour_path)
      mask = poly_to_mask(coords_list, width, height)

      if show_masked_dicoms:
        ax = plt.subplot(1,3,1)
        plt.imshow(dicom)
        ax.set_title('DICOM')

        ax = plt.subplot(1,3,2) 
        plt.imshow(mask)
        ax.set_title('Mask')

        ax = plt.subplot(1,3,3)
        plt.imshow(dicom)
        plt.imshow(mask, alpha=MASK_OVERLAY_ALPHA)
        ax.set_title('Overlay')

        plt.show()

      yield (dicom, mask)

if __name__ == '__main__':
  dicom_dir = join(DATA_DIR, 'dicoms')
  contour_dir = join(DATA_DIR, 'contourfiles')
  dicom_mask_tups = list(get_dicom_mask_tups(dicom_dir, contour_dir))
  pprint(dicom_mask_tups)
