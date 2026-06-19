import argparse
import json
import os
from glob import glob


def process_partition(source_json_pattern, dest_dir, partition_name):
    """
    Processes JSON files for a specific split (TRAIN or TEST) and converts them to COCO format.
    Uses the image UUID as the primary file identifier in the COCO JSON structure.
    """
    json_files = glob(source_json_pattern)
    if not json_files:
        print(f"No {partition_name} JSON files found matching pattern: {source_json_pattern}")
        # Create an empty instances file to avoid missing output for that partition
        output_path = os.path.join(dest_dir, "instances.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({"images": [], "annotations": [], "categories": []}, f, indent=4)
        return

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    categories_count = 0
    annotations_count = 0
    image_id_counter = 0

    # To handle image and annotation uniqueness *within this partition*
    # Key is now the Image UUID, as requested
    processed_images = {}  # image_uuid -> image_id

    # To ensure categories are unique across all JSONs being processed for this split
    category_name_to_id = {}

    for json_file in json_files:
        print(f"Processing {partition_name} part: {os.path.basename(json_file)}...")
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding {json_file}")
                continue

            content = data.get("content", [])

            for item in content:
                image_info = item.get("image", {})
                image_uuid = image_info.get("uuid")

                if not image_uuid:
                    continue

                image_width = image_info.get("width")
                image_height = image_info.get("height")

                # Check if image is already processed within this partition's set
                if image_uuid not in processed_images:
                    image_id = image_id_counter
                    processed_images[image_uuid] = image_id
                    image_id_counter += 1

                    coco_data["images"].append({
                        "id": image_id,
                        "file_name": image_uuid,
                        "width": image_width,
                        "height": image_height
                    })
                else:
                    image_id = processed_images[image_uuid]

                # Process annotations/labels
                labels = item.get("sampleLabels", [])
                for label in labels:
                    label_name = label.get("name")
                    if not label_name:
                        continue

                    # Build or retrieve category ID
                    if label_name not in category_name_to_id:
                        categories_count += 1
                        category_name_to_id[label_name] = categories_count
                        coco_data["categories"].append({
                            "id": categories_count,
                            "name": label_name,
                            "supercategory": ""
                        })

                    category_id = category_name_to_id[label_name]

                    # Bounding box extraction
                    bbox_data = label.get("boundingBox", {})
                    if not bbox_data:
                        continue

                    xmin = max(0, bbox_data.get("xmin"))
                    ymin = max(0, bbox_data.get("ymin"))
                    xmax = min(image_width - 1, bbox_data.get("xmax"))
                    ymax = min(image_height - 1, bbox_data.get("ymax"))

                    if None in (xmin, ymin, xmax, ymax):
                        continue

                    width = xmax - xmin
                    height = ymax - ymin

                    if width < 4 or height < 4:
                        continue

                    if xmin < 0 or ymin < 0 or xmax > image_width - 1 or ymax > image_height - 1 or width < 4 or height < 4:
                        raise ValueError("Invalid box, please add further transformations.")

                    annotations_count += 1
                    coco_data["annotations"].append({
                        "id": annotations_count,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [xmin, ymin, width, height],
                        "area": width * height,
                        "iscrowd": 0,
                        "segmentation": []
                    })

    # Save the COCO json
    output_path = os.path.join(dest_dir, "instances.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print("\n================================")
    print(f"✅ Transformation SUCCESS for {partition_name.upper()} split.")
    print(f"   Images: {len(coco_data['images'])}")
    print(f"   Annotations: {len(coco_data['annotations'])}")
    print(f"   Categories: {len(coco_data['categories'])}")
    print(f"   Saved to: {output_path}")
    print("================================\n")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Convert .pth or .ckpt model to onnx.",
        epilog=(
            "Example:\n"
            "  python convert_dsk_to_coco.py \\\n"
            "    -i /home/claudio/.makewise/dsk_cache/dataset/56d8e189-7b30-40f5-a949-274d87482c2d \\\n"
            "    -o /home/claudio/.makewise/dsk_cache/dataset/coco_format_dataset_56d8e189-7b30-40f5-a949-274d87482c2d"
        ),
    )
    parser.add_argument("-i", "--in_path", type=str, help="Path to the input dsk dataset.", required=True)
    parser.add_argument("-o", "--out_path", type=str, help="Path to the output coco dataset.", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    in_path = args.in_path
    out_path = args.out_path

    # 1. Process TRAIN split
    TRAIN_SOURCE_PATTERN = os.path.join(in_path, "TRAIN_*.json")
    process_partition(TRAIN_SOURCE_PATTERN, os.path.join(out_path, "train"), "TRAIN")

    # 2. Process TEST split
    TEST_SOURCE_PATTERN = os.path.join(in_path, "TEST_*.json")
    process_partition(TEST_SOURCE_PATTERN, os.path.join(out_path, "test"), "TEST")
