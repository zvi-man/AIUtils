from typing import List


class GtagConfig:
    working_dir: str = r''
    garbadge_dir: str = r''
    default_label: str = "AAAAAA"
    delimiter_pat: str = r"-|_"
    acceptable_file_types: List[str] = ["jpg", "bmp"]
    id_location_in_file_name: int = 2
    label_location_in_file_name: int = 3


