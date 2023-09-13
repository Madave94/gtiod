import argparse
import csv
import json
def parse_args():
    parser = argparse.ArgumentParser(description="Converts results file to kaggle compatible csv")
    parser.add_argument("input_file", type=str, help="input file containing the models output")
    parser.add_argument("output_file", type=str, help="output csv containing the results")

    args = parser.parse_args()

    return args

def vindrcxr_to_kaggle_submission_format(input, output):
    with open(input, "r") as f:
        results = json.load(f)

    intermediate = []

    for img_name, entries in results.items():
        img_name = img_name.split(".")[0]
        # if no results report empty prediction string
        if len(entries["bboxes"]) == 0:
            intermediate.append(
                [
                    img_name,
                    "14 1 0 0 1 1"
                ]
            )
        # if results create prediction string per results entry
        else:
            prep_entry = ""
            for bbox, label_id in zip(entries["bboxes"], entries["label_ids"]):
                prep_entry += "{} {} {} {} {} {} ".format(label_id, bbox[4], bbox[0], bbox[1], bbox[2], bbox[3])
            intermediate.append(
                [
                    img_name,
                    prep_entry
                ]
            )

    # write results to csv
    with open(output, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "PredictionString"])
        writer.writerows(intermediate)

def main():
    args = parse_args()

    vindrcxr_to_kaggle_submission_format(args.input_file, args.output_file)

if __name__ == '__main__':
    main()