#!/usr/bin/env python3


"""Pipeline for parsing DICOM and i-contour files"""


# Set to True for verbose output
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

# Batch size
BATCH_SIZE = 8

# Number of worker processes to use when loading batches asynchronously
NUM_WORKERS = 4


import argparse
import csv
import numpy as np
import logging
import multiprocessing as mp
from collections import Counter
from matplotlib import pyplot as plt
from os import listdir, getpid
from os.path import isfile, join
from pprint import pprint, pformat
from random import shuffle, seed
from threading import get_ident 


from parsing import parse_contour_file, parse_dicom_file, poly_to_mask


_log_format = '%(asctime)s : %(name)s : %(levelname)s : %(message)s'
logging.basicConfig(format=_log_format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

DICOM_DIR = join(DATA_DIR, 'dicoms')
CONTOUR_DIR = join(DATA_DIR, 'contourfiles')

def _get_cid_by_did(
    link_path=join(DATA_DIR, LINK_FNAME),
    dicom_key=DICOM_KEY,
    contour_key=CONTOUR_KEY,
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

def _get_path_by_id(base_path, get_id_func):
  logger.debug('base_path: %s' % base_path)
  path_by_id = {}
  for f in listdir(base_path):
    f_path = join(base_path, f)
    if not isfile(f_path):
      continue
    try:
      _id = get_id_func(f)
    except:
      logger.warn('Unable to parse id for file %s' % f)
      continue
    logger.debug('id: %s, f_path: %s' % (_id, f_path))
    if _id in path_by_id:
      logger.warn('duplicate _id %s' % _id)
      continue
    path_by_id[_id] = f_path
  return path_by_id

def _get_dicom_contour_path_tups(
    dicom_dir,
    contour_dir,
    randomize=False,
    random_seed=None,
):

  cid_by_did = _get_cid_by_did()
  
  rval = []
  for d_id in sorted(cid_by_did.keys()):
    c_id = cid_by_did[d_id]

    # read dicom paths
    dicom_path_base = join(dicom_dir, d_id)
    dicom_path_by_id = _get_path_by_id(
        dicom_path_base,
        lambda f: int(f.split('.')[0])
    )

    # read contour paths
    contour_path_base = join(contour_dir, c_id, 'i-contours')
    contour_path_by_id = _get_path_by_id(
        contour_path_base,
        lambda f: int(f.split('-')[2])
    )

    # pair up common ids
    dicom_ids = set(dicom_path_by_id.keys())
    contour_ids = set(contour_path_by_id.keys())
    common_ids = list(dicom_ids.intersection(contour_ids))
    logger.debug('common_ids: %s' % pformat(common_ids))

    for _id in sorted(common_ids):
      dicom_path = dicom_path_by_id[_id]
      contour_path = contour_path_by_id[_id]
      rval.append((dicom_path, contour_path))
  
  # shuffle
  if randomize:
    if random_seed is not None:
      logger.debug('setting random seed: %s' % random_seed)
      seed(random_seed)
    shuffle(rval)

  logger.debug('rval: %s' % pformat(rval))

  return rval

def _load_dicom_contour_paths(
    path_tups,
    show_masked_dicoms=False,
):
  for dicom_path, contour_path in path_tups:
    logger.debug('loading paths:\n\tdicom_path: %s\n\tcontour_path: %s' % (
      dicom_path, contour_path))

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

    #yield (np.random.randint(10, size=(2,3)), np.random.randint(10, size=(2,3)))
    yield (dicom, mask)
  
# Part 1
def get_dicom_mask_tups(
    dicom_dir=DICOM_DIR,
    contour_dir=CONTOUR_DIR,
):
  '''
  Function that takes in the directories containing the DICOM and contour files
  and provides an iterator over two-element tuples, where each two-element tuple
  consists of an image and its associated mask as Numpy arrays.
  '''

  path_tups = _get_dicom_contour_path_tups(dicom_dir, contour_dir)
  yield from _load_dicom_contour_paths(path_tups)

def _load_batch(path_tups):
  pid = getpid()
  logger.debug('worker %d loading batch of len %s' % (pid, len(path_tups)))
  gen = _load_dicom_contour_paths(path_tups)
  images, targets = [], []
  for image, target in gen:
    images.append(image)
    targets.append(target)
  logger.debug('worker %s batch loaded' % pid)
  return np.array(images), np.array(targets)

def _to_chunks(l, n):
  '''Yield successive n-sized chunks from l'''
  rval = []
  for i in range(0, len(l), n):
    rval.append(l[i:i + n])
  return rval

# Part 2
def get_batches(
    dicom_dir=DICOM_DIR,
    contour_dir=CONTOUR_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
):
  path_tups = _get_dicom_contour_path_tups(dicom_dir, contour_dir)
  path_tups_chunked = _to_chunks(path_tups, batch_size)

  pool = mp.Pool(num_workers)
  batches = pool.imap_unordered(
      _load_batch,
      path_tups_chunked,
  )
  for images, targets in batches:
    yield images, targets

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-a',
      help='Run part 1',
      action='store_true')
  parser.add_argument(
      '-b',
      help='Run part 2',
      action='store_true')
  args = parser.parse_args()

  if args.a:
    tups = get_dicom_mask_tups()
    for i, (dicom, mask) in enumerate(tups):
      logger.info('iteration %d, dicom.shape: %s, mask.shape: %s' % (
        i, dicom.shape, mask.shape))

  if args.b:
    batches = get_batches()
    try:
      i = 0
      while True:
        dicom, mask = next(batches)
        logger.info('iteration %d, dicom.shape: %s, mask.shape: %s' % (
          i, dicom.shape, mask.shape))
        import time; time.sleep(1)
        i += 1
    except StopIteration:
      pass

if __name__ == '__main__':
  main()
