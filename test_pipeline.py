import csv
import os
import pytest
import tempfile

from pipeline import (
    DICOM_KEY,
    CONTOUR_KEY,
    _get_cid_by_did,
    get_dicom_mask_tups,
    BatchFeeder
)

links = [
  ('patient_id_1', 'original_id_1'),
  ('patient_id_2', 'original_id_2')
]

@pytest.fixture
def link_path():
  _, file_path = tempfile.mkstemp()
  with open(file_path, 'w') as csv_file:
    fieldnames = [DICOM_KEY, CONTOUR_KEY]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for dicom_id, contour_id in links:
      writer.writerow({DICOM_KEY: dicom_id, CONTOUR_KEY: contour_id})

  yield file_path

  os.remove(file_path)

def test_get_cid_by_did(link_path):
  cid_by_did = _get_cid_by_did(link_path)
  for did, cid in links:
    assert cid_by_did[did] == cid

def test_get_dicom_mask_tups__smoke():
  tups = get_dicom_mask_tups()

def test_BatchFeeder__smoke():
  batch_feeder = BatchFeeder()
  [b for b in batch_feeder.get_next_batch()]


# TODO: more tests
