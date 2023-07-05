import os
import hashlib
from tqdm import tqdm


class FsUtils:
    @classmethod
    def remove_duplicate_files_from_dir(cls, dir_path: str) -> None:
        file_hashes = {}
        num_files_removed = 0
        for file_name in tqdm(os.listdir(dir_path)):
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path):
                file_hash = cls.get_file_hash(file_path)
                if file_hash in file_hashes:
                    os.remove(file_path)
                    num_files_removed += 1
                    print(f"file: {file_name} removed, it is the same as {os.path.basename(file_hashes[file_hash])}")
                else:
                    file_hashes[file_hash] = file_path

        print(f"Finished removing duplicates, Removed {num_files_removed}, left {len(file_hashes)}")

    @classmethod
    def get_file_hash(cls, file_path: str) -> str:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash


if __name__ == '__main__':
    DUPLICATE_DIR_PATH = r"D:\עבודה צבי\adversarial course ziv katzir"
    FsUtils.remove_duplicate_files_from_dir(DUPLICATE_DIR_PATH)
