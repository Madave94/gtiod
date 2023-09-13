import argparse
import csv
import os
from PIL import Image
import json

def add_root_dir_to_dataset_config(config, root_dir):
    """
        This function provides the root directory path and appends it in front of the image and annotations paths.
    """
    config.data.train.ann_file = root_dir + config.data.train.ann_file
    config.data.train.img_prefix = root_dir + config.data.train.img_prefix
    config.data.val.ann_file = root_dir + config.data.val.ann_file
    config.data.val.img_prefix = root_dir + config.data.val.img_prefix
    config.data.test.ann_file = root_dir + config.data.test.ann_file
    config.data.test.img_prefix = root_dir + config.data.test.img_prefix
    return config

def vindr_csv_to_coco_json(source_path, target_path):
    """
        This function takes the train.csv file provided from the kaggle source and converts it into coco format,
        the coco format can then be used for training.

        For class 14 (no finding) no annotation is added. Each image contains the original file_name and a newly assigned
        image_id for each different radiologist as well as rad_id which corresponds to the source rad_id.
    """
    # check and open source image
    csv_source = source_path + "train.csv"
    folder_source = source_path + "train/"
    assert os.path.exists(csv_source), "Path {} does not exist".format(csv_source)
    data = []
    with open(csv_source, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        print(next(csvreader))
        for row in csvreader:
            data.append(row)

    # create a dictionary of image_id : {rad_id: coco_image_id}
    # the original image_id will become the file_name
    images_to_rad = dict()
    counter = 1
    for file_name, _, _, rad_id, _, _, _, _ in data:
        if file_name in images_to_rad:
            if rad_id in images_to_rad[file_name]:
                continue
            else:
                images_to_rad[file_name][rad_id] = counter
                counter += 1
        else:
            images_to_rad[file_name] = {rad_id: counter}
            counter += 1

    # create coco categories and mapping that can be used later
    categories = [{"name": "Aortic enlargement", "id": 0, "supercategory": None},{"name": "Atelectasis", "id": 1, "supercategory": None},
                  {"name": "Calcification", "id": 2, "supercategory": None}, {"name": "Cardiomegaly", "id": 3, "supercategory": None},
                  {"name": "Consolidation", "id": 4, "supercategory": None}, {"name": "ILD", "id": 5, "supercategory": None},
                  {"name": "Infiltration", "id": 6, "supercategory": None}, {"name": "Lung Opacity", "id": 7, "supercategory": None},
                  {"name": "Nodule/Mass", "id":8, "supercategory": None}, {"name": "Other lesion", "id": 9, "supercategory": None},
                  {"name": "Pleural effusion", "id": 10, "supercategory": None}, {"name": "Pleural thickening", "id": 11, "supercategory": None},
                  {"name": "Pneumothorax", "id": 12, "supercategory": None}, {"name": "Pulmonary fibrosis", "id": 13, "supercategory": None}]

    # create coco annotations by going through all annotations
    counter = 1
    annotations = []
    for file_name, _, class_id, rad_id, x_min, y_min, x_max, y_max in data:
        class_id = int(class_id)
        if class_id == 14:
            continue
        x_min, y_min, x_max, y_max = float(x_min), float(y_min), float(x_max), float(y_max)
        image_id = images_to_rad[file_name][rad_id]
        annotations.append(
            {"id": counter,
             "image_id": image_id,
             "category_id": class_id,
             "segmentation": [[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]],
             "area": (x_max-x_min) * (y_max-y_min),
             "bbox": [x_min, y_min, x_max-x_min, y_max-y_min],
             "iscrowd": 0}
        )
        counter += 1

    # create coco image with width, height, id, file_name and rad_id
    images = []
    for file_name, rads in images_to_rad.items():
        img = Image.open(folder_source + file_name + ".png")
        width = img.width
        height = img.height
        for rad, id in rads.items():
            images.append(
                {"id": id,
                 "width": width,
                 "height": height,
                 "rad_id": rad,
                 "file_name": file_name + ".png"}
            )

    # combine the coco file and store it at target_path
    annotations_all = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(target_path, "w") as f:
        json.dump(annotations_all, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Call this function to convert the VinDr-CXR csv to coco json")
    parser.add_argument("source_path", help="The source folder.")
    parser.add_argument("target_path", help="The location where to put the coco json.")

    args = parser.parse_args()

    vindr_csv_to_coco_json(args.source_path, args.target_path)