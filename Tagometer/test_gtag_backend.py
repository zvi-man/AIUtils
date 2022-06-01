# Gtag backend Features
# 1. Browse through objects
# 2. Label objects - only selected images
# 3. Remove label and restore default label
# 4. statistics of - num_images, tagged images, unique labels, num images tagged
# 5. read file with predicted labels and offer the user to select them
# 6. set images as garbage and move the files to garbage dir
from random import randint
from typing import List, Dict
import os
import pytest
import logging

from Tagometer.gtag_config import GtagConfig
from Tagometer.gtag_backend import GtagBackEnd

logging.basicConfig(format='%(asctime)s - %(levelname)s, %(threadName)s, %(name)s, "%(message)s"',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d, %H:%M:%S')


def create_empty_file(file_path: str) -> None:
    with open(file_path, "w") as f:
        f.write("empty file")


def generate_filename(obj_id: str, label: str) -> str:
    video_num = "4"
    frame_num = str(randint(0, 1000000))
    return video_num + '-' + "_".join([frame_num, obj_id, label]) + ".jpg"


def init_working_dir(working_dir: str, id_labels_dict: Dict[str, List[str]]) -> None:
    for obj_id in id_labels_dict:
        for label in id_labels_dict[obj_id]:
            file_name = generate_filename(obj_id, label)
            file_path = os.path.join(working_dir, file_name)
            create_empty_file(file_path)


class TestGtagBackEnd(object):
    WORKING_DIR = r"working_dir"
    BASIC_WORKING_DIR_STRUCTURE = {
        "1": [GtagConfig.default_label],
        "2": [GtagConfig.default_label] * 2,
        "3": [GtagConfig.default_label] * 3,
        "4": [GtagConfig.default_label] * 4,
        "5": [GtagConfig.default_label] * 5
    }

    @pytest.fixture(scope="class", autouse=True)
    def _create_working_dir(self):
        os.mkdir(self.WORKING_DIR)
        yield
        os.rmdir(self.WORKING_DIR)

    def test_init_success(self):
        init_working_dir(self.WORKING_DIR, self.BASIC_WORKING_DIR_STRUCTURE)
        gtag = GtagBackEnd(self.WORKING_DIR)
        a = 1

    def test_browse_through_objects(self):
        pass

    def test_label_object(self):
        pass

    def test_label_object_subset(self):
        pass

    def test_unlabel_object(self):
        pass

    def test_move_object_to_garbage(self):
        pass
