#!/usr/bin/env python3


"""Pipeline for parsing DICOM and i-contour files"""


# Set to True for verbose output
DEBUG = False

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

# Number of image/target pairs to load per batch
BATCH_SIZE = 8


import argparse
import csv
import numpy as np
import logging
import multiprocessing as mp
import time
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

#########
# Part 1
#########

def get_dicom_mask_tups(
    dicom_dir=DICOM_DIR,
    contour_dir=CONTOUR_DIR,
):
  '''Returns an iterator over image/mask tuples loaded from the given paths'''

  path_tups = _get_dicom_contour_path_tups(dicom_dir, contour_dir)
  yield from _load_dicom_contour_paths(path_tups)


#########
# Part 2
#########

class BatchFeeder(object):
  '''Load batches one by one in a separate process'''

  def __init__(self,
      dicom_dir=DICOM_DIR,
      contour_dir=CONTOUR_DIR,
      batch_size=BATCH_SIZE,
  ):
    self._batch_num = 0
    self._q = mp.Manager().Queue()

    path_tups = _get_dicom_contour_path_tups(
        dicom_dir, contour_dir, randomize=True)
    self._path_tups_chunked = np.array_split(path_tups, batch_size)

    # pre-load first batch
    self._have_next_batch = self._load_next_batch()

  def _load_batch(self, path_tups):
    '''Load the files contained in the given paths'''
    pid = getpid()
    logger.debug('worker %d loading batch of len %s' % (pid, len(path_tups)))
    gen = _load_dicom_contour_paths(path_tups)
    images, targets = [], []
    for image, target in gen:
      images.append(image)
      targets.append(target)
    logger.debug('worker %s batch loaded' % pid)
    self._q.put((np.array(images), np.array(targets)))

  def _load_next_batch(self):
    '''Start a process to load the next batch'''
    if self._batch_num < len(self._path_tups_chunked):
      path_tups = self._path_tups_chunked[self._batch_num]
      p = mp.Process(target=self._load_batch, args=(path_tups,))
      p.start()
      self._batch_num += 1
    return self._batch_num < len(self._path_tups_chunked)

  def get_next_batch(self):
    '''Start loading next batch, and return previously loaded batch'''
    while self._have_next_batch:
      self._have_next_batch = self._load_next_batch()
      yield self._q.get()
    # we should have one more result waiting to be yielded
    yield self._q.get()
    raise StopIteration

###################
# Helper functions
###################

def _get_cid_by_did(
    link_path,
    dicom_key=DICOM_KEY,
    contour_key=CONTOUR_KEY,
):
  '''Read links between dicoms and contours'''

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
  '''Return a dict of id: path under the given base path'''
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
    link_path=join(DATA_DIR, LINK_FNAME),
    randomize=False,
    random_seed=None,
):
  '''Returns a list of dicom/contour path tuples'''

  cid_by_did = _get_cid_by_did(link_path)
  
  path_tups = []
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
    common_ids = sorted(list(dicom_ids.intersection(contour_ids)))
    logger.debug('common_ids: %s' % common_ids)

    for _id in common_ids:
      dicom_path = dicom_path_by_id[_id]
      contour_path = contour_path_by_id[_id]
      path_tups.append((dicom_path, contour_path))
  
  # shuffle
  if randomize:
    if random_seed is not None:
      logger.debug('setting random seed: %s' % random_seed)
      seed(random_seed)
    shuffle(path_tups)

  logger.debug('path_tups:\n%s' % pformat(path_tups))

  return path_tups

def _make_masked_dicom(i, dicom, mask, show=False, save=False):
  if not (show or save):
    return

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

  if show:
    logger.info('Displaying masked dicom %s' % i)
    plt.show()

  if save:
    filename = 'tmp_%04d.png' % i
    logger.info('Saving masked dicom %s' % filename)
    plt.savefig(filename)

def _load_dicom_contour_paths(
    path_tups,
    show_masked_dicoms=False,
    save_masked_dicoms=False,
):
  '''Returns an iterator over image/mask tuples loaded from the given paths'''

  for i, (dicom_path, contour_path) in enumerate(path_tups):
    logger.debug('loading paths:\n\tdicom_path: %s\n\tcontour_path: %s' % (
      dicom_path, contour_path))

    # read dicom data
    dcm_dict = parse_dicom_file(dicom_path)
    dicom = dcm_dict['pixel_data']
    width, height = dicom.shape

    # read contour data
    coords_list = parse_contour_file(contour_path)
    mask = poly_to_mask(coords_list, width, height)

    _make_masked_dicom(i, dicom, mask, show_masked_dicoms, save_masked_dicoms)

    yield (dicom, mask)

def run_part_1():
  tups = get_dicom_mask_tups()
  for i, (dicom, mask) in enumerate(tups):
    logger.info('iteration %d, dicom.shape: %s, mask.shape: %s' % (
      i, dicom.shape, mask.shape))

def run_part_2(save_masked_dicoms=False):
  batch_feeder = BatchFeeder()
  for i, (dicom, mask) in enumerate(batch_feeder.get_next_batch()):
    logger.info('iteration %d, dicom.shape: %s, mask.shape: %s' % (
      i, dicom.shape, mask.shape))

    N = dicom.shape[0]
    for j in range(N):
      d = dicom[j,:,:]
      m = mask[j,:,:]
      _make_masked_dicom(i*N + j, d, m, save=save_masked_dicoms)

    # simulate training
    logger.info('training...')
    time.sleep(1)

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
    run_part_1()

  if args.b:
    run_part_2()

if __name__ == '__main__':
  main()
