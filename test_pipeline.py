import csv
import os
import pytest
import tempfile

from pipeline import _get_cid_by_did, DICOM_KEY, CONTOUR_KEY

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

# TODO: more tests
