import os
import xml.etree.ElementTree as ET

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager


DATASET_ROOT = os.environ.get("DETECTRON2_DATASETS", "datasets")


def load_filtered_voc_instances(
    name: str, dirname: str, split: str, classnames: str
):
    """
    Load Pascal VOC detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    is_shots = "shot" in name
    if is_shots:
        fileids = {}
        split_dir = os.path.join(DATASET_ROOT, "vocsplit")
        if "seed" in name:
            shot = name.split("_")[-2].split("shot")[0]
            seed = int(name.split("_seed")[-1])
            split_dir = os.path.join(split_dir, "seed{}".format(seed))
        else:
            shot = name.split("_")[-1].split("shot")[0]
        for cls in classnames:
            with PathManager.open(
                os.path.join(
                    split_dir, "box_{}shot_{}_train.txt".format(shot, cls)
                )
            ) as f:
                fileids_ = np.loadtxt(f, dtype=np.str).tolist()
                if isinstance(fileids_, str):
                    fileids_ = [fileids_]
                fileids_ = [
                    fid.split("/")[-1].split(".jpg")[0] for fid in fileids_
                ]
                fileids[cls] = fileids_
    else:
        if '2007_2012' in dirname:
            with PathManager.open(
                os.path.join(dirname.replace('2007_2012', '2007'), "ImageSets", "Main", split + ".txt")
            ) as f:
                fileids_2007 = np.loadtxt(f, dtype=np.str)
            with PathManager.open(
                os.path.join(dirname.replace('2007_2012', '2012'), "ImageSets", "Main", split + ".txt")
            ) as f:
                fileids_2012 = np.loadtxt(f, dtype=np.str)
            fileids = []
            for f in fileids_2007: fileids.append((2007, f))
            for f in fileids_2012: fileids.append((2012, f))
        else:
            with PathManager.open(
                os.path.join(dirname, "ImageSets", "Main", split + ".txt")
            ) as f:
                fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    if is_shots:
        for cls, fileids_ in fileids.items():
            dicts_ = []
            for fileid in fileids_:
                year = "2012" if "_" in fileid else "2007"
                dirname = os.path.join(DATASET_ROOT, "VOC{}".format(year))
                anno_file = os.path.join(
                    dirname, "Annotations", fileid + ".xml"
                )
                jpeg_file = os.path.join(
                    dirname, "JPEGImages", fileid + ".jpg"
                )

                tree = ET.parse(anno_file)

                for obj in tree.findall("object"):
                    r = {
                        "file_name": jpeg_file,
                        "image_id": fileid,
                        "height": int(tree.findall("./size/height")[0].text),
                        "width": int(tree.findall("./size/width")[0].text),
                    }
                    cls_ = obj.find("name").text
                    if cls != cls_:
                        continue
                    bbox = obj.find("bndbox")
                    bbox = [
                        float(bbox.find(x).text)
                        for x in ["xmin", "ymin", "xmax", "ymax"]
                    ]
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0

                    instances = [
                        {
                            "category_id": classnames.index(cls),
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                        }
                    ]
                    r["annotations"] = instances
                    dicts_.append(r)
            if len(dicts_) > int(shot):
                dicts_ = np.random.choice(dicts_, int(shot), replace=False)
            dicts.extend(dicts_)
    else:
        for fileid in fileids:
            if isinstance(fileid, (tuple, list)):
                anno_file = os.path.join(dirname.replace('2007_2012', str(fileid[0])), "Annotations", fileid[1] + ".xml")
                jpeg_file = os.path.join(dirname.replace('2007_2012', str(fileid[0])), "JPEGImages", fileid[1] + ".jpg")
            else:
                anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
                jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

            tree = ET.parse(anno_file)

            r = {
                "file_name": jpeg_file,
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            }
            instances = []

            for obj in tree.findall("object"):
                cls = obj.find("name").text
                if not (cls in classnames):
                    continue
                bbox = obj.find("bndbox")
                bbox = [
                    float(bbox.find(x).text)
                    for x in ["xmin", "ymin", "xmax", "ymax"]
                ]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                instances.append(
                    {
                        "category_id": classnames.index(cls),
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                    }
                )
            r["annotations"] = instances
            dicts.append(r)
    return dicts


def register_meta_pascal_voc(
    name, metadata, dirname, split, year, keepclasses, sid
):
    if keepclasses.startswith("base_novel"):
        thing_classes = metadata["thing_classes"][sid]
    elif keepclasses.startswith("base"):
        thing_classes = metadata["base_classes"][sid]
    elif keepclasses.startswith("novel"):
        thing_classes = metadata["novel_classes"][sid]

    DatasetCatalog.register(
        name,
        lambda: load_filtered_voc_instances(
            name, dirname, split, thing_classes
        ),
    )

    MetadataCatalog.get(name).set(
        thing_classes=thing_classes,
        dirname=dirname,
        year=year,
        split=split,
        base_classes=metadata["base_classes"][sid],
        novel_classes=metadata["novel_classes"][sid],
    )
    



# PASCAL VOC categories
PASCAL_VOC_ALL_CATEGORIES = {
    1: [
        "aeroplane",
        "bicycle",
        "boat",
        "bottle",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
        "bird",
        "bus",
        "cow",
        "motorbike",
        "sofa",
    ],
    2: [
        "bicycle",
        "bird",
        "boat",
        "bus",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
        "aeroplane",
        "bottle",
        "cow",
        "horse",
        "sofa",
    ],
    3: [
        "aeroplane",
        "bicycle",
        "bird",
        "bottle",
        "bus",
        "car",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "train",
        "tvmonitor",
        "boat",
        "cat",
        "motorbike",
        "sheep",
        "sofa",
    ],
}

PASCAL_VOC_NOVEL_CATEGORIES = {
    1: ["bird", "bus", "cow", "motorbike", "sofa"],
    2: ["aeroplane", "bottle", "cow", "horse", "sofa"],
    3: ["boat", "cat", "motorbike", "sheep", "sofa"],
}

PASCAL_VOC_BASE_CATEGORIES = {
    1: [
        "aeroplane",
        "bicycle",
        "boat",
        "bottle",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
    ],
    2: [
        "bicycle",
        "bird",
        "boat",
        "bus",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
    ],
    3: [
        "aeroplane",
        "bicycle",
        "bird",
        "bottle",
        "bus",
        "car",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "train",
        "tvmonitor",
    ],
}



def _get_pascal_voc_fewshot_instances_meta():
    ret = {
        "thing_classes": PASCAL_VOC_ALL_CATEGORIES,
        "novel_classes": PASCAL_VOC_NOVEL_CATEGORIES,
        "base_classes": PASCAL_VOC_BASE_CATEGORIES,
    }
    return ret

    

# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root="datasets"):
    # SPLITS = [
    #     ("voc_2007_trainval", "VOC2007", "trainval"),
    #     ("voc_2007_train", "VOC2007", "train"),
    #     ("voc_2007_val", "VOC2007", "val"),
    #     ("voc_2007_test", "VOC2007", "test"),
    #     ("voc_2012_trainval", "VOC2012", "trainval"),
    #     ("voc_2012_train", "VOC2012", "train"),
    #     ("voc_2012_val", "VOC2012", "val"),
    # ]
    # for name, dirname, split in SPLITS:
    #     year = 2007 if "2007" in name else 2012
    #     register_pascal_voc(name, os.path.join(root, dirname), split, year)
    #     MetadataCatalog.get(name).evaluator_type = "pascal_voc"

    # register meta datasets
    METASPLITS = [
        # ("voc_2007_trainval_base1", "VOC2007", "trainval", "base1", 1),
        # ("voc_2007_trainval_base2", "VOC2007", "trainval", "base2", 2),
        # ("voc_2007_trainval_base3", "VOC2007", "trainval", "base3", 3),
        # ("voc_2012_trainval_base1", "VOC2012", "trainval", "base1", 1),
        # ("voc_2012_trainval_base2", "VOC2012", "trainval", "base2", 2),
        ("pascal_voc_train_split_1", "VOC2007_2012", "trainval", "base1", 1),
        ("pascal_voc_train_split_2", "VOC2007_2012", "trainval", "base2", 2),
        ("pascal_voc_train_split_3", "VOC2007_2012", "trainval", "base3", 3),
        # ("voc_2007_trainval_all1", "VOC2007", "trainval", "base_novel_1", 1),
        # ("voc_2007_trainval_all2", "VOC2007", "trainval", "base_novel_2", 2),
        # ("voc_2007_trainval_all3", "VOC2007", "trainval", "base_novel_3", 3),
        # ("voc_2012_trainval_all1", "VOC2012", "trainval", "base_novel_1", 1),
        # ("voc_2012_trainval_all2", "VOC2012", "trainval", "base_novel_2", 2),
        # ("voc_2012_trainval_all3", "VOC2012", "trainval", "base_novel_3", 3),
        ("voc_2007_test_base1", "VOC2007", "test", "base1", 1),
        ("voc_2007_test_base2", "VOC2007", "test", "base2", 2),
        ("voc_2007_test_base3", "VOC2007", "test", "base3", 3),
        ("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1),
        ("voc_2007_test_novel2", "VOC2007", "test", "novel2", 2),
        ("voc_2007_test_novel3", "VOC2007", "test", "novel3", 3),
        ("voc_2007_test_all1", "VOC2007", "test", "base_novel_1", 1),
        ("voc_2007_test_all2", "VOC2007", "test", "base_novel_2", 2),
        ("voc_2007_test_all3", "VOC2007", "test", "base_novel_3", 3),
    ]

    # register small meta datasets for fine-tuning stage
    for prefix in [ "novel"]: # "all",
        for sid in range(1, 4):
            for shot in [1, 2, 3, 5, 10]:
                for year in [2007]: # 2012
                    for seed in [0]: 
                        seed = "" if seed == 0 else "_seed{}".format(seed)
                        name = "voc_{}_trainval_{}{}_{}shot{}".format(
                            year, prefix, sid, shot, seed
                        )
                        dirname = "VOC{}".format(year)
                        img_file = "{}_{}shot_split_{}_trainval".format(
                            prefix, shot, sid
                        )
                        keepclasses = (
                            "base_novel_{}".format(sid)
                            if prefix == "all"
                            else "novel{}".format(sid)
                        )
                        METASPLITS.append(
                            (name, dirname, img_file, keepclasses, sid)
                        )

    for name, dirname, split, keepclasses, sid in METASPLITS:
        year = 2007 if "2007" in name else 2012
        register_meta_pascal_voc(
            name,
            _get_pascal_voc_fewshot_instances_meta(),
            os.path.join(root, dirname),
            split,
            year,
            keepclasses,
            sid,
        )
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
        
register_all_pascal_voc(root=os.environ.get("DETECTRON2_DATASETS", "datasets"))
    # D1 = DatasetCatalog.get('pascal_voc_train_split_1')
    # D2 = DatasetCatalog.get('pascal_voc_train_split_2')
    # D3 = DatasetCatalog.get('pascal_voc_train_split_3')

if __name__ == "__main__":
    from lib.categories import ALL_CLS_DICT
    x1 = ALL_CLS_DICT['pascal_voc_train_split_3']
    x2 = MetadataCatalog.get('voc_2007_test_all3').thing_classes
    print(1)

    pass

# voc_2007_trainval_novel2_3shot
# voc_2007_test_novel1