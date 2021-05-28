import json
import glob
import os
import argparse

from tqdm import tqdm
from PIL import Image


def create_image_list(dataset_dir, output_dir, dataset):
  """
  :param dataset_dir: path to the PPS ground-truths file for the data split
  :param output_dir: directory where the images.json file will be stored
  :param dataset: dataset name ('Cityscapes' or 'Pascal')

  :return:
  """
  print("Creating images list...")
  images_list = list()

  # Get all filenames in the GT directory
  filenames = [file for file in glob.glob(dataset_dir + "/*")]
  if dataset == 'Cityscapes':
    filenames.extend([file for file in glob.glob(dataset_dir + "/*/*")])

  for filename in tqdm(filenames):
    if filename.endswith(str('.tif')):
      image_dict = dict()
      file_name_gt = os.path.basename(filename)

      # Set names for file_name and image_id
      if dataset == 'Cityscapes':
        file_name = file_name_gt.replace('_gtFinePanopticParts.tif', '_gtFine_leftImg8bit.png')
        image_id = file_name_gt.replace('_gtFinePanopticParts.tif', '')
      else:
        file_name = file_name_gt.replace('.tif', '.png')
        image_id = file_name_gt.replace('.tif', '')
      image_dict['file_name'] = file_name
      image_dict['id'] = image_id

      # Open gt image and store image dimensions
      img = Image.open(filename)
      image_dict['width'], image_dict['height'] = img.size[0:2]

      images_list.append(image_dict)

  images_dict = {'images': images_list}

  # Save images.json file
  output_path = os.path.join(output_dir, 'images.json')
  with open(output_path, 'w') as fp:
    json.dump(images_dict, fp)

  print("Created images list and stored at {}.".format(output_path))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="Creates an images.json file for the Cityscapes Panoptic Parts or Pascal Panoptic Parts dataset."
  )

  parser.add_argument('dataset_dir', type=str,
                      help="path to the PPS ground-truths file for the data split")
  parser.add_argument('output_dir', type=str,
                      help="directory where the images.json file will be stored")
  parser.add_argument('dataset', type=str,
                      help="dataset name ('Cityscapes' or 'Pascal')")
  args = parser.parse_args()

  create_image_list(args.dataset_dir,
                    args.output_dir,
                    dataset=args.dataset)
