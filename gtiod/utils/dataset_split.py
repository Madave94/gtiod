import argparse
import os
import json

def split_train_val_vindr(source, target_train, target_val, val_portion):
    assert 0 < val_portion < 1, "Validation amount must be between 0 and 1 not {}".format(val_portion)
    assert os.path.exists(source), "{} does not exist.".format(source)
    with open(source, "r") as f:
        all = json.load(f)
    file_names = set()
    for image in all["images"]:
        file_names.add(image["file_name"])
    file_names = sorted(file_names)
    val_index = int((len(file_names) * val_portion))
    val_file_names = file_names[:val_index]
    train_file_names = file_names[val_index:]
    val_image_ids = set()
    val_images = []
    train_image_ids = set()
    train_images = []
    for image in all["images"]:
        if image["file_name"] in val_file_names:
            val_image_ids.add(image["id"])
            val_images.append(image)
        if image["file_name"] in train_file_names:
            train_image_ids.add(image["id"])
            train_images.append(image)
    val_annotations = []
    train_annotations = []
    for annotation in all["annotations"]:
        if annotation["image_id"] in val_image_ids:
            val_annotations.append(annotation)
        if annotation["image_id"] in train_image_ids:
            train_annotations.append(annotation)
    with open(target_train, "w") as f:
        json.dump({"images": train_images, "annotations": train_annotations, "categories": all["categories"]}, f)
    with open(target_val, "w") as f:
        json.dump({"images": val_images, "annotations": val_annotations, "categories": all["categories"]}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Call this function to split the VinDr-CXR coco json into two parts.")
    parser.add_argument("source_path", help="The source annotation.")
    parser.add_argument("target_path_train", help="The target annotation file.")
    parser.add_argument("target_path_val", help="The target annotation file.")
    parser.add_argument("val_portion", help="validation part of the dataset e.g. 0.2 for 20%", type=float)

    args = parser.parse_args()

    split_train_val_vindr(args.source_path, args.target_path_train, args.target_path_val, args.val_portion)