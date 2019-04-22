import glob
import os
import xml.etree.ElementTree as ET

from PIL import Image

"""
Convert a Yolo dataset format to a PASCAL VOC dataset.
"""

TO_CONVERT_PATH = "/home/coac/Pictures/datasets/YOLO-dataset/"
OUTPUT_DIRECTORY = "dataset/train/"


def write_pascal_xml(
    file_path_xml,
    detections,
    image_width,
    image_height,
    image_filename="cat.png",
    image_folder="images",
    image_file_path="home/coac/Pictures/dataset/images/cat.jpeg",
    label_name="cat",
):
    annotation = ET.Element("annotation")
    folder = ET.SubElement(annotation, "folder")
    folder.text = image_folder

    filename = ET.SubElement(annotation, "filename")
    filename.text = image_filename

    path = ET.SubElement(annotation, "path")
    path.text = image_file_path

    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(image_width)
    height = ET.SubElement(size, "height")
    height.text = str(image_height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    for detection in detections:
        xmin, xmax, ymin, ymax = detection
        object = ET.SubElement(annotation, "object")
        name = ET.SubElement(object, "name")
        name.text = label_name
        pose = ET.SubElement(object, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(object, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(object, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(object, "bndbox")
        xmin_el = ET.SubElement(bndbox, "xmin")
        xmin_el.text = str(xmin)
        ymin_el = ET.SubElement(bndbox, "ymin")
        ymin_el.text = str(ymin)
        xmax_el = ET.SubElement(bndbox, "xmax")
        xmax_el.text = str(xmax)
        ymax_el = ET.SubElement(bndbox, "ymax")
        ymax_el.text = str(ymax)

    mydata = ET.tostring(annotation)
    myfile = open(file_path_xml, "wb")
    myfile.write(mydata)


def parse_yolo_file(file_path):
    file = open(file_path, "r")
    data = file.read()
    file.close()

    if data == "":
        return []

    lines = [line for line in data.split("\n") if line != ""]
    detections = []
    for line in lines:
        parsed = [float(d) for d in line.split(" ")]
        label = parsed[0]
        x_norm = parsed[1]
        y_norm = parsed[2]
        sizeX_norm = parsed[3]
        sizeY_norm = parsed[4]

        detections.append((label, x_norm, y_norm, sizeX_norm, sizeY_norm))

    return detections


def yolo_to_kitti_format(x_norm, y_norm, sizeX_norm, sizeY_norm):
    xmin_norm = x_norm - sizeX_norm / 2
    ymin_norm = y_norm - sizeY_norm / 2
    xmax_norm = x_norm + sizeX_norm / 2
    ymax_norm = y_norm + sizeY_norm / 2

    x = x_norm * width
    y = y_norm * height
    sizeX = sizeX_norm * width
    sizeY = sizeY_norm * height

    xmin = xmin_norm * width
    xmax = xmax_norm * width
    ymin = ymin_norm * height
    ymax = ymax_norm * height

    return int(xmin), int(xmax), int(ymin), int(ymax)


if __name__ == "__main__":
    images_path = glob.glob(TO_CONVERT_PATH + "*.png") + glob.glob(TO_CONVERT_PATH + "*.jpg")
    print(len(images_path), "images to convert...")

    for image_path in images_path:
        im = Image.open(image_path)
        rgb_im = im.convert("RGB")

        width, height = im.size

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        annotation_path = os.path.join(os.path.dirname(image_path), image_name + ".txt")

        detections = parse_yolo_file(annotation_path)

        detections_kitti = []
        for detection in detections:
            label, x_norm, y_norm, sizeX_norm, sizeY_norm = detection
            xmin, xmax, ymin, ymax = yolo_to_kitti_format(x_norm, y_norm, sizeX_norm, sizeY_norm)
            if xmax > width:
                xmax = width
            if ymax > height:
                ymax = height
            detections_kitti.append((xmin, xmax, ymin, ymax))

        images_dir = OUTPUT_DIRECTORY + "images/"
        annotations_dir = OUTPUT_DIRECTORY + "annotations/"
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        if not os.path.exists(annotations_dir):
            os.makedirs(annotations_dir)

        rgb_im.save(images_dir + image_name + ".jpeg")
        write_pascal_xml(
            annotations_dir + image_name + ".xml",
            detections=detections_kitti,
            image_width=width,
            image_height=height,
            image_file_path=image_path,
            label_name="enemy",
            image_filename=image_name + ".jpeg",
        )
