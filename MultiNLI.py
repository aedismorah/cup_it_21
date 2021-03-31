import wget
import jsonlines
import zipfile
import os
import pandas as pd
import shutil

class Printer():
  def __init__(self, silent=False):
    self.silent = silent
  
  def print(self, text):
    if self.silent == False:
      print(text)

def download_and_unzip_multinli(download_folder='', unzip_folder='', silent=False):
  '''
  Usage example:
    !pip install wget
    !pip install jsonlines
    train_data, m_dev_data, mism_dev_data = download_and_unzip_multinli('', '')

  Input: download_folder - Folder to download snli_1.0.zip.
         unzip_folder - Folder to unzip dataset

  Return (tuple of lists): train_data, matched_dev_data, mismatched_dev_data
  '''
  pr = Printer(silent=silent)

  url = 'https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip'
  pr.print(f'Downloading multinli_1.0.zip to {download_folder}')
  file = wget.download(url, out=download_folder)

  from_folder = os.path.join(download_folder, 'multinli_1.0.zip')
  pr.print(f'Unzipping {from_folder} to {unzip_folder}')
  with zipfile.ZipFile(from_folder, 'r') as zip_ref:
      zip_ref.extractall(unzip_folder)

  train_data = []
  matched_dev_data = []
  mismatched_dev_data = []


  from_folder_train = os.path.join(unzip_folder, 'multinli_1.0/multinli_1.0_train.jsonl')
  pr.print(f'Reading {from_folder_train}')
  with jsonlines.open(from_folder_train) as reader:
      for obj in reader:
        train_data.append(obj)

  from_folder_dev_matched = os.path.join(unzip_folder, 'multinli_1.0/multinli_1.0_dev_matched.jsonl')
  pr.print(f'Reading {from_folder_dev_matched}')
  with jsonlines.open(from_folder_dev_matched) as reader:
      for obj in reader:
        matched_dev_data.append(obj)

  from_folder_dev_mismatched = os.path.join(unzip_folder, 'multinli_1.0/multinli_1.0_dev_mismatched.jsonl')
  pr.print(f'Reading {from_folder_dev_mismatched}')
  with jsonlines.open(from_folder_dev_mismatched) as reader:
      for obj in reader:
        mismatched_dev_data.append(obj)

  shutil.rmtree('__MACOSX')
  shutil.rmtree('multinli_1.0')
  os.remove('multinli_1.0.zip')

  pr.print('Done')

  return train_data, matched_dev_data, mismatched_dev_data

def pandas_decorator(func_get_data):
  '''
  Usage:
    !pip install wget
    !pip install jsonlines
    download_and_unzip_multinli_pd = pandas_decorator(download_and_unzip_multinli)
    train_df, m_dev_df, mism_dev_df = download_and_unzip_multinli_pd('', '')
  '''
  def make_df(data_list):
    return pd.DataFrame(list(map(lambda x: x.values(), data_list)), columns=list(data_list[0].keys()))

  def wrapper(*args, **kwargs):
    train_data, dev_data, test_data = func_get_data(*args, **kwargs)

    if 'silent' in kwargs.keys():
      if kwargs['silent'] == False:
        print('Making train_df')
    else:
      print('Making train_df')
    train_df = make_df(train_data)

    if 'silent' in kwargs.keys():
      if kwargs['silent'] == False:
        print('Making matched_dev_df')
    else:
      print('Making matched_dev_df')
    dev_df = make_df(dev_data)

    if 'silent' in kwargs.keys():
      if kwargs['silent'] == False:
        print('Making mismatched_dev_df')
    else:
      print('Making mismatched_dev_df')
    test_df = make_df(test_data)
    return train_df, dev_df, test_df

  return wrapper