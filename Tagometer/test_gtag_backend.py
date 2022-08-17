# Gtag backend Features
# 1. Browse through objects
# 2. Label objects - only selected images
# 3. Remove label and restore default label
# 4. statistics of - num_images, tagged images, unique labels, num images tagged
# 5. read file with predicted labels and offer the user to select them
# 6. set images as garbage and move the files to garbage dir
from random import randint
from typing import List, Dict, Generator
import os
import pytest
import logging
import shutil

from KMUtils.Tagometer.gtag_config import GtagConfig
from KMUtils.Tagometer.gtag_backend import GtagBackEnd

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
    TEST_LABEL = "abc123"
    SUBSET_TO_LABEL1 = [2]
    SUBSET_TO_LABEL2 = [1]
    BASIC_WORKING_DIR_STRUCTURE = {
        "1": [GtagConfig.default_label],
        "2": [GtagConfig.default_label] * 2,
        "3": [GtagConfig.default_label] * 3,
        "4": [GtagConfig.default_label] * 4,
        "5": [GtagConfig.default_label] * 5
    }

    @pytest.fixture(autouse=True)
    def _init_working_dir(self) -> Generator:
        os.mkdir(self.WORKING_DIR)
        init_working_dir(self.WORKING_DIR, self.BASIC_WORKING_DIR_STRUCTURE)
        yield
        shutil.rmtree(self.WORKING_DIR)

    def _get_labeled_files(self) -> List[str]:
        return [file_name for file_name in os.listdir(self.WORKING_DIR)
                if self.TEST_LABEL in file_name]

    def test_init_success(self) -> None:
        gtag = GtagBackEnd(self.WORKING_DIR)
        assert gtag.num_of_labeled_files == 0
        assert gtag.total_num_of_files == 15
        assert gtag.get_num_unique_labels() == 0

    def test_browse_through_objects(self) -> None:
        gtag = GtagBackEnd(self.WORKING_DIR)
        assert gtag.prev_object() is False
        assert gtag.next_object() is True
        assert gtag.next_object() is True
        assert gtag.get_current_object_id() == 3
        assert gtag.prev_object() is True
        assert gtag.get_current_object_id() == 2
        assert gtag.prev_object() is True
        assert gtag.get_current_object_id() == 1
        assert gtag.next_object() is True
        assert gtag.next_object() is True
        assert gtag.next_object() is True
        assert gtag.next_object() is True
        assert gtag.next_object() is False

    def test_label_object(self) -> None:
        gtag = GtagBackEnd(self.WORKING_DIR)
        gtag.next_object()
        gtag.next_object()
        assert gtag.get_current_object_label() == GtagConfig.default_label
        gtag.set_current_object_label(self.TEST_LABEL)
        assert gtag.get_current_object_label() == self.TEST_LABEL
        relabeled_files = self._get_labeled_files()
        assert len(relabeled_files) == 3

    def test_label_object_subset(self) -> None:
        gtag = GtagBackEnd(self.WORKING_DIR)
        gtag.next_object()
        gtag.next_object()
        old_file_name1 = gtag._get_current_object().file_list[2].file_name
        old_file_name2 = gtag._get_current_object().file_list[1].file_name
        new_file_name1 = old_file_name1.replace(GtagConfig.default_label, self.TEST_LABEL)
        new_file_name2 = old_file_name2.replace(GtagConfig.default_label, self.TEST_LABEL)
        gtag.set_current_object_label(self.TEST_LABEL, self.SUBSET_TO_LABEL1)
        assert new_file_name1 in os.listdir(self.WORKING_DIR)
        relabeled_files = self._get_labeled_files()
        assert len(relabeled_files) == 1
        # Test relabel a different subset of this object
        gtag.set_current_object_label(self.TEST_LABEL, self.SUBSET_TO_LABEL2)
        assert old_file_name1 in os.listdir(self.WORKING_DIR)
        assert new_file_name1 not in os.listdir(self.WORKING_DIR)
        assert new_file_name2 in os.listdir(self.WORKING_DIR)
        relabeled_files = self._get_labeled_files()
        assert len(relabeled_files) == 1

    def test_un_label_object(self) -> None:
        gtag = GtagBackEnd(self.WORKING_DIR)
        gtag.set_current_object_label(self.TEST_LABEL)
        gtag.un_label_current_object()
        labeled_files = self._get_labeled_files()
        assert len(labeled_files) == 0

    def test_unique_labels_count(self) -> None:
        pass

    def test_num_labeled_files(self) -> None:
        pass

    def test_total_num_files(self) -> None:
        pass

    def test_move_object_to_garbage(self) -> None:
        gtag = GtagBackEnd(self.WORKING_DIR)
        gtag.move_current_object_to_garbage()

