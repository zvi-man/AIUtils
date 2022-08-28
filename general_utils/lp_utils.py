import os
import re
from typing import List, Set

# Constants
DEFAULT_LABEL: str = "AAAAAA"
DEFAULT_DELIMITER_PAT: str = r"-|_|\."
ACCEPTABLE_FILE_TYPES: List[str] = ["jpg", "bmp", "png"]
ID_LOCATION_IN_FILE_NAME: int = 2
LABEL_LOCATION_IN_FILE_NAME: int = 3


class LPUtils(object):
    def __init__(self, delimiter_pat: str = DEFAULT_DELIMITER_PAT,
                 id_location_in_file_name: int = ID_LOCATION_IN_FILE_NAME,
                 label_location_in_file_name: int = LABEL_LOCATION_IN_FILE_NAME,
                 acceptable_file_types: List[str] = ACCEPTABLE_FILE_TYPES,
                 default_label: str = DEFAULT_LABEL):
        self.delimiter_pat = delimiter_pat
        self.id_location_in_file_name = id_location_in_file_name
        self.label_location_in_file_name = label_location_in_file_name
        self.acceptable_file_types = acceptable_file_types
        self.default_label = default_label

    def get_label_from_file_name(self, file_name: str) -> str:
        old_name_fields = re.split(self.delimiter_pat, file_name)
        return old_name_fields[self.label_location_in_file_name]

    def get_id_from_file_name(self, file_name: str) -> int:
        old_name_fields = re.split(self.delimiter_pat, file_name)
        id_str = old_name_fields[self.id_location_in_file_name]
        id_int = -1 if id_str == "UN" else int(id_str)
        return id_int

    def create_new_labeled_file_name(self, old_name: str, label: str) -> str:
        old_label = self.get_label_from_file_name(old_name)
        if old_name.count(old_label) != 1:
            raise NotImplementedError("File name should have only one occurrence of label")
        return old_name.replace(old_label, label)

    def is_file_labeled(self, file_name: str) -> bool:
        label = self.get_label_from_file_name(file_name)
        return label != self.default_label

    def get_labeled_tracklet_ids(self, lp_dir_path: str) -> Set[int]:
        labeled_tracklet_ids = set()
        for file_name in os.scandir(lp_dir_path):
            if os.path.splitext(file_name.name)[1] in self.acceptable_file_types:
                if self.is_file_labeled(file_name.name):
                    labeled_tracklet_ids.add(self.get_id_from_file_name(file_name.name))
        return labeled_tracklet_ids


