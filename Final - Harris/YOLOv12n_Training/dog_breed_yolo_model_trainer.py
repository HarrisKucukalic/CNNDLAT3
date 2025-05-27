# -*- coding: utf-8 -*-
"""
Code sourced from https://www.kaggle.com/code/vineetmahajan/dog-breed-detection
"""

from IPython.display import display, clear_output
from ipywidgets import Button, HBox, Label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import comet_ml
import wandb
from ultralytics import YOLO
import torch
import cv2
import PIL.Image as Image
import os
import shutil
import pathlib
import sys
import yaml
import xmltodict
import zipfile
import json
import yaml
import torch
from pathlib import Path

if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define dataset paths

    ROOT_DIR = pathlib.Path(r"C:\projects\CNNDLAT3\Final\YOLOv12n_Training")
    ANNOTATIONS_PATH = ROOT_DIR / 'annotations'
    IMAGES_PATH = ROOT_DIR / 'images'

    # Define paths for saving processed data
    NEW_DATA_PATH = ROOT_DIR / "data"
    NEW_ANNOTATIONS_PATH = NEW_DATA_PATH / "annotations"
    SLIDES_PATH = ROOT_DIR / "slides"

    # Create necessary directories
    os.makedirs(NEW_ANNOTATIONS_PATH, exist_ok=True)
    os.makedirs(SLIDES_PATH, exist_ok=True)

    # Change working directory
    os.chdir(ROOT_DIR)

    # Set random seed
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Set device to GPU to increase speed

    num_devices=1

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        device = [i for i in range(num_devices)]
        device_name = torch.cuda.get_device_name()
    elif torch.backends.mps.is_available():
        device = "mps"
        device_name = "mps"
    else:
        device = "cpu"
        device_name = "cpu"


    # Dataset
    TEST_TRAIN_SPLIT = 0.15
    VALIDATION_TRAIN_SPLIT = 0.15
    # Model Vars
    IMAGE_SIZE = 640
    # Training
    PROJECT_NAME = "dog_breed_detection"
    DEVICE = device
    BATCH_SIZE = 16
    EPOCHS = 200
    # Check device being used
    print(f"Using {device_name} as the Backend.")
    print(f"Number of Devices: {num_devices}")

    # Unzip Stanford Dogbreed detector
    with zipfile.ZipFile('Dog Breed Dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('.')


    # Set up subdirectories for each of the 120 dog breeds. Print statements used to test if this has occured properly
    breed_dir_name = [
        breed
        for breed in sorted(os.listdir('images/Images'))
        if not breed.startswith(".") and os.path.isdir((os.path.join('images/Images', breed)))
    ]
    print(f"Number of breeds: {len(breed_dir_name)}")
    print(f"First 5 breeds: {breed_dir_name[:5]}")



    # Initialise dataset DataFrame - uses file paths to reference the images and anotations
    dataset_df = pd.DataFrame(columns=["breed", "image_path", "annotation_path"])

    # Iterate through each breed directory
    for i, breed_dir in enumerate(breed_dir_name):
        breed_name = " ".join(breed_dir.replace("_", "-").split("-")[1:]).title()

        breed_images_dir_path = Path("images/Images") / breed_dir
        breed_annotations_dir_path = Path("annotations/Annotation") / breed_dir

        breed_images_name = [
            image
            for image in sorted(os.listdir(breed_images_dir_path))
            if not image.startswith(".") and image.endswith((".jpg", ".jpeg", ".png"))
        ]
        breed_annotations_name = [
            image.split(".")[0]
            for image in breed_images_name
        ]

        breed_images_path = [
            (breed_images_dir_path / image).as_posix()  # Convert Path object to string
            for image in breed_images_name
            if (breed_images_dir_path / image).is_file()  # Check if the file exists
        ]

        breed_annotations_path = [
            (breed_annotations_dir_path / annotation).as_posix()  # Convert Path object to string
            for annotation in breed_annotations_name
            if (breed_annotations_dir_path / annotation).is_file()  # Check if the file exists
        ]

        dataset_df = pd.concat([dataset_df, pd.DataFrame({
            "breed": breed_name,
            "image_path": breed_images_path,
            "annotation_path": breed_annotations_path
        })])

        if i % 10 == 0:
            print(f"Loading... {int(i / 1.20)}% done")

    # Display first few rows
    print(dataset_df.head())

    def read_txt(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content

    print(read_txt(dataset_df.iloc[0]['annotation_path']))

    dataset_df_path = NEW_DATA_PATH / "dataset_df.csv"
    dataset_df = dataset_df.sort_values(by=["breed", "image_path"])
    dataset_df.to_csv(dataset_df_path, index=False)

    dataset_df = pd.read_csv(dataset_df_path)
    print(dataset_df.head())



    breeds = dataset_df["breed"].unique()
    breeds_dict = {i: breed for i, breed in enumerate(breeds)}

    breed_id_dict = {breed: i for i, breed in breeds_dict.items()}

    # Converts Pascal VOC annotations to YOLO format
    new_annotations = []
    for i, annotation_path in enumerate(dataset_df["annotation_path"]):
        annotation_path = pathlib.Path(annotation_path)
        new_annotation_path = NEW_ANNOTATIONS_PATH / annotation_path.parent.name
        os.makedirs(new_annotation_path, exist_ok=True)
        annotation_name = annotation_path.name

        annotation_data = xmltodict.parse(annotation_path.read_text())
        image_w, image_h = (
            int(annotation_data["annotation"]["size"]["width"]),
            int(annotation_data["annotation"]["size"]["height"])
        )
        final_data = ""
        objects = annotation_data["annotation"]["object"]
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            breed = obj["name"]
            xmin, ymin, xmax, ymax = (
                int(obj["bndbox"]["xmin"]),
                int(obj["bndbox"]["ymin"]),
                int(obj["bndbox"]["xmax"]),
                int(obj["bndbox"]["ymax"])
            )
            obj_h = ymax - ymin
            obj_w = xmax - xmin
            class_id = breed_id_dict[breed.replace("_", " ").replace("-", " ").title()]
            x, y, w, h = (
                (xmin + obj_w/2) / image_w,
                (ymin + obj_h/2) / image_h,
                (obj_w) /image_w ,
                (obj_h) / image_h
            )

            final_data += f"{class_id} {x} {y} {w} {h}\n"

        if i%2000==0:
            print(f"{i} done")
        new_annotation_path = new_annotation_path / (annotation_name+".txt")
        new_annotation_path.write_text(final_data)
        new_annotations.append(new_annotation_path)

    dataset_df["new_annotation_path"] = new_annotations
    dataset_df.to_csv(dataset_df_path, index=False)

    dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)


    def read_txt(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content

    # Each dataframe has the directory locations of the data for easier dataset manipulation
    test_df = dataset_df.sample(frac=TEST_TRAIN_SPLIT)
    train_df = dataset_df.drop(test_df.index)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df_path = NEW_DATA_PATH / "train_df.csv"
    test_df_path = NEW_DATA_PATH / "test_df.csv"

    train_df.to_csv(train_df_path, index=False)
    test_df.to_csv(test_df_path, index=False)

    val_split = VALIDATION_TRAIN_SPLIT
    val_df = train_df.sample(frac=val_split)
    train_df = train_df.drop(val_df.index)

    a,b,c = len(train_df), len(val_df), len(test_df)

    print(f"Train set has {a} images")
    print(f"Validation set has {b} images")
    print(f"Test set has {c} images")

    train_dir = NEW_DATA_PATH / "train"
    val_dir = NEW_DATA_PATH / "val"
    test_dir = NEW_DATA_PATH / "test"

    dataset_details = {
        "path": str(NEW_DATA_PATH),
        "train": "train",
        "val": "val",
        "test": "test",

        "names": breeds_dict
    }
    # Set dataset yaml path
    dataset_yaml_path = ROOT_DIR / "detection/dataset.yaml"
    os.makedirs(dataset_yaml_path.parent, exist_ok=True)

    with open(dataset_yaml_path, "w") as f:
        yaml.dump(dataset_details, f, default_flow_style=False)

    CLASS_NAMES = list(breeds_dict.values())
    NUM_CLASSES = len(CLASS_NAMES)

    # load model weights to be used to drain
    model_name = "yolov12n.pt"
    version = "v3.0"
    experiment_name = f"{model_name}-{version}"

    model_file_path = os.path.join(os.getcwd(), "yolov12n.pt")
    # create model
    model = YOLO(model_file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting training...")
    # Hyperparameter setting - workers used to avoid excessive memory usage, patience used for early stopping.
    model.train(
        data=dataset_yaml_path,
        workers=1,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        project=PROJECT_NAME,

        name=experiment_name,

        exist_ok=True,
        patience = 15,
        save_period=10
    )
