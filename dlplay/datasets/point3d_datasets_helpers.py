from enum import Enum

from dlplay.datasets.modelnet import ModelNet
from dlplay.datasets.s3dis import S3DIS
from dlplay.datasets.shapenet import ShapeNet

# from dlplay.datasets.scannet_simple import ScanNet   # this is a simpler version of the ScanNet dataset not depending on torch-points3d
from dlplay.datasets.scannet import ScanNet

from dlplay.paths import DATA_DIR
from typing import Optional, Callable


class PointDatasetType(Enum):
    ModelNet10 = "ModelNet10"  # 10 categories classification, only classification
    ModelNet40 = "ModelNet40"  # 40 categories classification, only classification
    S3DIS = "S3DIS"  # 13 semantic classes (12 + clutter) in 6 areas across 3 buildings, only segmentation
    ShapeNet = "ShapeNet"  # 16 shape categories, both segmentation and classification
    ScanNet = (
        "ScanNet"  # semantic segmentation (commonly 20 classes); test split unlabeled
    )


class PointTask(Enum):
    Classification = "Classification"
    Segmentation = "Segmentation"
    Detection = "Detection"
    Tracking = "Tracking"
    Reconstruction = "Reconstruction"
    Generation = "Generation"


def get_datasets(
    dataset_type: PointDatasetType,
    task: PointTask,  # to double check
    dataset_force_reload: bool = False,
    transform: Optional[Callable] = None,
    pre_transform: Optional[Callable] = None,
    only_test: bool = False,
    categories: list | None = None,  # only used for Shapenet at present time
    process_workers: int = 1,  # only used for ScanNet at present time
):
    """
    Get the train and test datasets for the given dataset name.
    """
    train_dataset, test_dataset, validation_dataset = None, None, None
    get_train_dataset = not only_test

    if dataset_type == PointDatasetType.ModelNet10:
        if task != PointTask.Classification:
            raise ValueError(f"Task {task} not supported for {dataset_type}")
        if get_train_dataset:
            train_dataset = ModelNet(
                root=f"{DATA_DIR}/datasets/ModelNet10",
                name="10",
                train=True,
                force_reload=dataset_force_reload,
                transform=transform,
                pre_transform=pre_transform,
            )
        test_dataset = ModelNet(
            root=f"{DATA_DIR}/datasets/ModelNet10",
            name="10",
            train=False,
            force_reload=dataset_force_reload,
            transform=transform,
            pre_transform=pre_transform,
        )

    elif dataset_type == PointDatasetType.ModelNet40:
        if task != PointTask.Classification:
            raise ValueError(f"Task {task} not supported for {dataset_type}")
        if get_train_dataset:
            train_dataset = ModelNet(
                root=f"{DATA_DIR}/datasets/ModelNet40",
                name="40",
                train=True,
                force_reload=dataset_force_reload,
                transform=transform,
                pre_transform=pre_transform,
            )
        test_dataset = ModelNet(
            root=f"{DATA_DIR}/datasets/ModelNet40",
            name="40",
            train=False,
            force_reload=dataset_force_reload,
            transform=transform,
            pre_transform=pre_transform,
        )

    elif dataset_type == PointDatasetType.S3DIS:
        if task != PointTask.Segmentation:
            raise ValueError(f"Task {task} not supported for {dataset_type}")
        if get_train_dataset:
            train_dataset = S3DIS(
                root=f"{DATA_DIR}/datasets/S3DIS",
                test_area=6,
                train=True,
                force_reload=dataset_force_reload,
                transform=transform,
                pre_transform=pre_transform,
            )
        test_dataset = S3DIS(
            root=f"{DATA_DIR}/datasets/S3DIS",
            test_area=6,
            train=False,
            force_reload=dataset_force_reload,
            transform=transform,
            pre_transform=pre_transform,
        )

    elif dataset_type == PointDatasetType.ShapeNet:
        if task != PointTask.Segmentation:
            raise ValueError(f"Task {task} not supported for {dataset_type}")
        if get_train_dataset:
            train_dataset = ShapeNet(
                root=f"{DATA_DIR}/datasets/ShapeNet",
                categories=categories,
                split="trainval",
                force_reload=dataset_force_reload,
                transform=transform,
                pre_transform=pre_transform,
            )
        test_dataset = ShapeNet(
            root=f"{DATA_DIR}/datasets/ShapeNet",
            categories=categories,
            split="test",
            force_reload=dataset_force_reload,
            transform=transform,
            pre_transform=pre_transform,
        )

    elif dataset_type == PointDatasetType.ScanNet:
        if task != PointTask.Segmentation:
            raise ValueError(f"Task {task} not supported for {dataset_type}")
        # Common practice: use 'train' and 'val'; 'test' has no labels.
        if get_train_dataset:
            train_dataset = ScanNet(
                root=f"{DATA_DIR}/datasets/ScanNet",
                split="train",
                transform=transform,
                pre_transform=pre_transform,
                process_workers=process_workers,
                force_reload=dataset_force_reload,
            )
        # Prefer 'val' for evaluation; switch to 'test' only if you really need the test split.
        test_split = "val" if not only_test else "test"
        test_dataset = ScanNet(
            root=f"{DATA_DIR}/datasets/ScanNet",
            split=test_split,
            transform=transform,
            pre_transform=pre_transform,
            process_workers=process_workers,
            force_reload=dataset_force_reload,
        )
        validation_dataset = ScanNet(
            root=f"{DATA_DIR}/datasets/ScanNet",
            split="val",
            transform=transform,
            pre_transform=pre_transform,
            process_workers=process_workers,
            force_reload=dataset_force_reload,
        )

    else:
        raise ValueError(f"Dataset {dataset_type} not found")

    return train_dataset, test_dataset, validation_dataset


def get_categories_names(dataset_type: PointDatasetType):
    if dataset_type == PointDatasetType.ModelNet10:
        return [
            "bathtub",
            "bed",
            "chair",
            "desk",
            "dresser",
            "monitor",
            "night_stand",
            "sofa",
            "table",
            "toilet",
        ]
    elif dataset_type == PointDatasetType.ModelNet40:
        return [
            "airplane",
            "bathtub",
            "bed",
            "bench",
            "bookshelf",
            "bottle",
            "bowl",
            "car",
            "chair",
            "cone",
            "cup",
            "curtain",
            "desk",
            "door",
            "dresser",
            "flower_pot",
            "glass_box",
            "guitar",
            "keyboard",
            "lamp",
            "laptop",
            "mantel",
            "monitor",
            "night_stand",
            "person",
            "piano",
            "plant",
            "radio",
            "range_hood",
            "sink",
            "sofa",
            "stairs",
            "stool",
            "table",
            "tent",
            "toilet",
            "tv_stand",
            "vase",
            "wardrobe",
            "xbox",
        ]
    elif dataset_type == PointDatasetType.ShapeNet:
        category_ids = {
            "Airplane": "02691156",
            "Bag": "02773838",
            "Cap": "02954340",
            "Car": "02958343",
            "Chair": "03001627",
            "Earphone": "03261776",
            "Guitar": "03467517",
            "Knife": "03624134",
            "Lamp": "03636649",
            "Laptop": "03642806",
            "Motorbike": "03790512",
            "Mug": "03797390",
            "Pistol": "03948459",
            "Rocket": "04099429",
            "Skateboard": "04225987",
            "Table": "04379243",
        }
        return list(category_ids.keys())
    elif dataset_type == PointDatasetType.ScanNet:
        # Standard 20-class ScanNet semantic labels
        return [
            "wall",
            "floor",
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "door",
            "window",
            "bookshelf",
            "picture",
            "counter",
            "desk",
            "curtain",
            "refrigerator",
            "shower curtain",
            "toilet",
            "sink",
            "bathtub",
            "other furniture",
        ]
    else:
        raise ValueError(f"Dataset {dataset_type} not found")
