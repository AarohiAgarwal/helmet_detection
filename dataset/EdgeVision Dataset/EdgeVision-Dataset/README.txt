# EdgeVision Dataset

## Overview

The EdgeVision Dataset is a curated collection of annotated images designed for
helmet detection research in motorcycle safety and intelligent transportation
systems. The dataset was created by Kandarp Gajjar, Nancy Patel, Ridham Patel,
Krutika Patel, Chrisha Dabhi, Himani Trivedi, Mahesh Goyani,
and Chintan Bhatt.

## Dataset Contents

EdgeVision-Dataset/
    images/                                  2,392 JPEG images (Image_00001.jpg ...)
        Image_00001.jpg ...
    labels/
        voc_xml/                             Pascal VOC, one .xml per image
            Image_00001.xml ...
        yolo/                                YOLO, one .txt per image, normalized coords
            Image_00001.txt ...
            classes.txt                      class index order: BikeWithRider, NoHelmet, Helmet
        coco/
            annotations.json                 COCO format
            annotations.csv                  flat tabular export
    LICENSE.txt
    README.txt

## Annotation Format

Annotations are provided in three formats: Pascal VOC XML, YOLO, and COCO.

### Pascal VOC Format (labels/voc_xml/)

One .xml file per image. Each <object> element holds the class <name> and an
absolute-pixel <bndbox> (xmin, ymin, xmax, ymax).

### YOLO Format (labels/yolo/)

Each annotation file contains one object per line:

    <class_id> <x_center> <y_center> <width> <height>

All coordinates are normalized with respect to image dimensions (values in [0, 1]).

Class mapping:

    0 - BikeWithRider
    1 - NoHelmet
    2 - Helmet

### COCO Format (labels/coco/)

annotations.json follows the COCO specification (images, annotations,
categories). Bounding boxes are stored as absolute pixels [x, y, width, height].
annotations.csv is a flat tabular export of the same annotations.

Note: YOLO format uses 0-based class IDs (0, 1, 2). The COCO format uses 1-based
category IDs (1, 2, 3) as per the COCO specification. The class order is
identical across all three formats.

## Dataset Statistics

* Total Images:      2392
* Total Annotations: 8275

### Average Image Resolution

    1154 x 1411 pixels (width x height)

## Recommended Citation

If you use this dataset in your research, please cite:

    Gajjar, Kandarp; Patel, Nancy; Patel, Ridham; Patel, Krutika; Dabhi, Chrisha;
    Trivedi, Himani; Goyani, Mahesh; Bhatt, Chintan (2026), "EdgeVision Dataset",
    Mendeley Data, V1, doi: 10.17632/j82bnw7gsr.1

## License

This dataset is licensed under the Creative Commons Attribution 4.0
International License (CC BY 4.0).

You are free to:
    Share  - copy and redistribute the material in any medium or format
    Adapt  - remix, transform, and build upon the material for any purpose

Under the following terms:
    Attribution - You must give appropriate credit, provide a link to the
                  license, and indicate if changes were made.

For full license text, see LICENSE.txt or visit:
    https://creativecommons.org/licenses/by/4.0/

## Contact

For questions regarding the dataset, please contact the corresponding authors.
