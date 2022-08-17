import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Dict
import re

from KMUtils.Tagometer.gtag_config import GtagConfig


class GtagBackEndException(Exception):
    pass


def get_label_from_file_name(file_name: str) -> str:
    old_name_fields = re.split(GtagConfig.delimiter_pat, file_name)
    return old_name_fields[GtagConfig.label_location_in_file_name]


def get_id_from_file_name(file_name: str) -> int:
    old_name_fields = re.split(GtagConfig.delimiter_pat, file_name)
    id_str = old_name_fields[GtagConfig.id_location_in_file_name]
    id_int = -1 if id_str == "UN" else int(id_str)
    return id_int


def create_new_labeled_file_name(old_name: str, label: str) -> str:
    old_label = get_label_from_file_name(old_name)
    if old_name.count(old_label) != 1:
        raise NotImplementedError("File name should have only one occurrence of label")
    return old_name.replace(old_label, label)


def rename_file(old_file_name: str, new_file_name: str, containing_dir: Optional[str] = None) -> None:
    if containing_dir is not None:
        old_file_name = os.path.join(containing_dir, old_file_name)
        new_file_name = os.path.join(containing_dir, new_file_name)
    os.rename(old_file_name, new_file_name)


def move_file(old_file_path: str, new_dir: str) -> None:
    file_name = os.path.basename(old_file_path)
    new_file_path = os.path.join(new_dir, file_name)
    os.rename(old_file_path, new_file_path)


@dataclass
class LabelTrackedFile:
    file_name: str = ''
    file_dir: str = ''
    # The is_labeled param and obj_id is learned during the post_init function
    is_labeled: bool = field(init=False)
    obj_id: int = field(init=False)

    def __post_init__(self) -> None:
        label = get_label_from_file_name(self.file_name)
        self.obj_id = get_id_from_file_name(self.file_name)
        self.is_labeled = label != GtagConfig.default_label

    def label_file(self, label: str) -> None:
        new_file_name = create_new_labeled_file_name(self.file_name, label)
        rename_file(self.file_name, new_file_name, self.file_dir)
        self.file_name = new_file_name
        self.is_labeled = True

    def un_label_file(self) -> None:
        if self.is_labeled:
            unlabeled_file_name = create_new_labeled_file_name(self.file_name, GtagConfig.default_label)
            rename_file(self.file_name, unlabeled_file_name, self.file_dir)
            self.file_name = unlabeled_file_name
            self.is_labeled = False


@dataclass
class LabelledObject:
    id: int
    file_list: List[LabelTrackedFile]
    # The label is learned during the post_init function
    label: str = field(init=False, default=GtagConfig.default_label)

    def __post_init__(self) -> None:
        all_file_labels = [get_label_from_file_name(labeled_file.file_name) for labeled_file in self.file_list]
        all_file_labels_set = set(all_file_labels)
        if all_file_labels_set == {GtagConfig.default_label}:
            self.label = GtagConfig.default_label
            return
        if GtagConfig.default_label in all_file_labels_set:
            all_file_labels_set.remove(GtagConfig.default_label)
        if len(all_file_labels_set) != 1:
            raise GtagBackEndException(f"Error, current object id: {self.id} "
                                       f"has multiple labels: {all_file_labels_set}")
        self.label = all_file_labels_set.pop()

    def un_label_object(self) -> None:
        self.label = GtagConfig.default_label
        for labeled_file in self.file_list:
            labeled_file.un_label_file()

    def label_object(self, label: str, subset_to_label: Optional[List[int]] = None) -> None:
        self.un_label_object()
        for file_num, label_tracked_file in enumerate(self.file_list):
            if subset_to_label is None or file_num in subset_to_label:
                label_tracked_file.label_file(label)
        self.label = label

    def get_num_of_labeled_files(self) -> int:
        return len([labeled_file for labeled_file in self.file_list if labeled_file.is_labeled])


class GtagBackEnd(object):
    def __init__(self, working_dir: str = GtagConfig.working_dir):
        # TODO: add default label attribute
        self.idx: int = 0
        self.working_dir: str = working_dir
        self.total_num_of_files: int = 0
        self.num_of_labeled_files: int = 0
        self.obj_list: List[LabelledObject] = []
        self.unique_labels_set: Set[str] = set()
        self._init_obj_list()

    def _init_obj_list(self) -> None:
        obj_dict = self._sort_files_by_object()
        for object_id, labeled_file_list in obj_dict.items():
            labeled_object = LabelledObject(object_id, labeled_file_list)
            self.obj_list.append(labeled_object)
        self.obj_list.sort(key=lambda x: x.id)

    def _sort_files_by_object(self) -> Dict[int, List[LabelTrackedFile]]:
        obj_dict = dict()
        all_files = [file_name for file_name in os.listdir(self.working_dir)
                     if Path(file_name).suffix[1:] in GtagConfig.acceptable_file_types]
        self.total_num_of_files = len(all_files)
        for file_name in all_files:
            labeled_file = LabelTrackedFile(file_name, self.working_dir)
            if labeled_file.is_labeled:
                self.unique_labels_set.add(get_label_from_file_name(labeled_file.file_name))
                self.num_of_labeled_files += 1
            if labeled_file.obj_id not in obj_dict:
                obj_dict[labeled_file.obj_id] = [labeled_file]
            else:
                obj_dict[labeled_file.obj_id].append(labeled_file)
        return obj_dict

    def next_object(self) -> bool:
        if (self.idx + 1) == len(self.obj_list):
            return False
        self.idx += 1
        return True

    def prev_object(self) -> bool:
        if self.idx == 0:
            return False
        self.idx -= 1
        return True

    def _get_current_object(self) -> LabelledObject:
        return self.obj_list[self.idx]

    def get_current_object_id(self) -> int:
        return self._get_current_object().id

    def get_current_object_file_paths(self) -> List[str]:
        object_files_paths = []
        for label_tracked_file in self._get_current_object().file_list:
            file_path = os.path.join(label_tracked_file.file_dir, label_tracked_file.file_name)
            object_files_paths.append(file_path)
        return object_files_paths

    def move_current_object_to_garbage(self) -> None:
        # TODO: add track of total num of files + num of labeled files
        for file_path in self.get_current_object_file_paths():
            move_file(file_path, GtagConfig.garbage_dir)
        del self.obj_list[self.idx]

    def get_current_object_label(self) -> str:
        return self._get_current_object().label

    def set_current_object_label(self, label: str, subset_to_label: Optional[List[int]] = None) -> None:
        if label != GtagConfig.default_label:
            self.unique_labels_set.add(label)
        # TODO: add track of num of labeled files
        current_object = self._get_current_object()
        current_object.label_object(label, subset_to_label)

    def un_label_current_object(self) -> None:
        current_object = self._get_current_object()
        if current_object.label != GtagConfig.default_label:
            self.unique_labels_set.remove(current_object.label)
        current_object.un_label_object()

    def get_num_unique_labels(self) -> int:
        return len(self.unique_labels_set)

    def get_num_objects(self) -> int:
        return len(self.obj_list)

