import argparse
import json
import random

import mmcv
import numpy as np
import torch
import mmdet
from mmdet.apis import init_detector, inference_detector
from mmdet.core.mask.structures import bitmap_to_polygon
from tqdm import tqdm
import os
import imghdr
from pathlib import Path
from datetime import date
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Runs an inference on a folder or single image.")
    parser.add_argument("config_file", help="Path to config file.", type=str)
    parser.add_argument("checkpoint_file", help="Path to the used weights.", type=str)
    parser.add_argument("inference_path", help="Path to the image folder or the image.", type=str)

    parser.add_argument("--device", type=str, default="cpu",
                        help="specify the device to run the inference on, default is cpu. cuda:0 means the first gpu",)
    parser.add_argument("--plot_store_path", type=str, default=None,
                        help="path to store inference images with boxes or segmentation masks. If no path is given, plots are not created.")
    parser.add_argument("--results_path", type=str, default=None,
                        help="path for saving a results.json. If no path is given, results are not stored.")
    parser.add_argument("--valid_image_postfixes", nargs="+", type=str, help="list all valid image types",
                        default=["gif", "pbm", "pgm", "ppm", "tiff", "jpeg", "bmp", "png", "jpg", "tif"],)
    parser.add_argument("--confidence_threshold", type=float, default=0.3, help="Minimum confidence to be included in the results")

    # additional argument to filter files (inference on a reduced set, from the selected folder)
    parser.add_argument("--filter", type=str, default=None,
                        help="Add a filter, only files that include part of the path are included")
    # additional argument to select only a few paths to create a random sample
    parser.add_argument("--random_sample", type=int, default=None, help="Use this parameter to create a small random"
                                                                        "set instead of all samples.")

    # additional configs for creating VLP inference
    parser.add_argument("--vlp_inference", action="store_true", help="activate this to store the files in vlp inference format instead")
    parser.set_defaults(vlp_inference=False)
    parser.add_argument("--dlcv_format", action="store_true", help="activate this to store the files in dlcv inference format instead")
    parser.set_defaults(vlp_inference=False)
    parser.add_argument("--direct_inference", action="store_true", help="activate this to store inferenced files directly")
    parser.set_defaults(skip_existing=False)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_version", type=str, default=None)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    config_file = args.config_file
    checkpoint_file = args.checkpoint_file
    path_to_images = args.inference_path
    assert os.path.exists(path_to_images), "Path {} for inference images does not exist. Cannot make inference without images".format(path_to_images)
    device = args.device
    valid_postfixes = [x.lower() for x in args.valid_image_postfixes]

    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)

    results = dict()
    metadata = dict()
    if os.path.isdir(path_to_images):
        path_obj = Path(path_to_images)
        paths = path_obj.rglob("*")
        if args.filter is not None:
            paths = list(paths)
            paths = [path for path in paths if args.filter in path]
        counter = 0
        for img_path in tqdm(paths):
            img_path = str(img_path)
            if img_path.split(".")[-1].lower() in valid_postfixes:
                try:
                    result = inference_detector(model, img_path)
                except:
                    print(f"Error in image {img_path}. Skipping this image.")
                    continue
                # each box (x y x y confidence)
                # create a list of results here with confidence and classes depending on confidence threshold
                # do the same for segmentation masks
                bboxes, segms, labels = extract_results(result, args.confidence_threshold)
                width, height = Image.open(img_path).size
                label_names = [model.CLASSES[int(label)] for label in labels]
                results[os.path.basename(img_path)] = {"bboxes": bboxes,
                                     "segms": segms,
                                     "label_ids": labels,
                                     "label_names": label_names}
                metadata[os.path.basename(img_path)] = {"width": width, "height": height, "img_path": img_path}
                if args.plot_store_path != None:
                    model.show_result(img_path, result, out_file= os.path.join(args.plot_store_path, os.path.basename(img_path)))
                counter += 1
                if args.random_sample and counter >= args.random_sample:
                    break
                if args.direct_inference:
                    if args.results_path is not None and args.vlp_inference:
                        store_vlp_format(results, metadata, args)
                        results = dict()
                        metadata = dict()
    else:
        img_path = path_to_images
        img_or_not = imghdr.what(img_path)
        if img_or_not in valid_postfixes:
            result = inference_detector(model, img_path)
            # each box (x y x y confidence)
            # create a list of results here with confidence and classes depending on confidence threshold
            # do the same for segmentation masks
            bboxes, segms, labels = extract_results(result, args.confidence_threshold)
            label_names = [model.CLASSES[int(label)] for label in labels]
            results[os.path.basename(img_path)] = {"bboxes": bboxes,
                                                   "segms": segms,
                                                   "label_ids": labels,
                                                   "label_names": label_names}
            if args.plot_store_path != None:
                model.show_result(img_path, result, out_file=os.path.join(args.plot_store_path, os.path.basename(img_path)))
        else:
            assert False, "{} is neither a valid image nor a folder".format(path_to_images)

    # store results
    if args.results_path is not None and args.vlp_inference:
        store_vlp_format(results, metadata, args)
    elif args.results_path is not None and args.dlcv_format:
        store_dlcv_format(results, metadata, args)
    elif args.results_path is not None:
        with open(args.results_path, "w") as f:
            json.dump(results, f)
    else:
        print("No results path selected. Not storing results.")

def extract_results(result, score_thr):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)
    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]
    polygons = []
    if segms is not None:
        for i, mask in enumerate(segms):
            contours, _ = bitmap_to_polygon(mask)
            for contour in contours:
                polygon = []
                for point in contour:
                    polygon.append(str(point[0]))
                    polygon.append(str(point[1]))
                polygons.append(polygon)
    new_bboxes = []
    for bbox in bboxes:
        new_bbox = []
        for entry in list(bbox):
            new_bbox.append(str(entry))
        new_bboxes.append(new_bbox)
    labels = [str(label) for label in list(labels)]
    return new_bboxes, polygons, labels

def store_dlcv_format(results, metadata, args):
    result_all = []
    for file_name, result in results.items():
        for i in range( len(result["bboxes"]) ):
            x,y,x2,y2,score = [float(entry) for entry in result["bboxes"][i]]
            # to coco format xywh and class start at 1 because 0 is saved for abckground
            result_all.append(
                {
                    "file_name": file_name,
                    "bbox": [x,y,x2-x,y2-y],
                    "category_id": int(result["label_ids"][i])+1,
                    "score": score
                }
            )
    with open(os.path.join(args.results_path), "w") as f:
        json.dump(result_all,f)

def store_vlp_format(results, metadata_all, args):
    # extract model and args meta data
    today = str(date.today())
    software_version = "pytorch_{}_mmdetection_{}".format(torch.__version__, mmdet.__version__)
    model_name = args.model_name
    version = args.model_version

    layout_element_id = 1
    # iterate through results and create inference files
    for image_name, contents in results.items():
        try:
            doc_name = metadata_all[image_name]["img_path"].split("/")[-2] # extract folder name, but only of one folder ahead
            json_name = image_name.split(".")[0] + ".json"
            metadata = {
                "document_name": image_name,
                "date": today,
                "software_version": software_version,
                "model_name": model_name,
                "version": version,
                "image_width": metadata_all[image_name]["width"],
                "image_height": metadata_all[image_name]["height"]
            }
            data_list = []
            for idx in range(len(contents["bboxes"])):
                data_list.append(
                    {
                        "layout_element_id": layout_element_id,
                        "bbox": contents["bboxes"][idx],
                        "segm": contents["segms"][idx],
                        "label_ids": contents["label_ids"][idx],
                        "label_names": contents["label_names"][idx]
                    }
                )
                layout_element_id += 1
            os.makedirs(os.path.join(args.results_path, doc_name), exist_ok=True)
            with open(os.path.join(args.results_path, doc_name, json_name), "w") as f:
                json.dump(
                    {
                        "metadata": metadata,
                        "data": data_list
                    }, f)
        except:
            print(f"Error in creating json for {image_name}. Skipping this image.")

if __name__ == '__main__':
    main()