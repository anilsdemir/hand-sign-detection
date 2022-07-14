import os
import string

from common.config import FILE_DIR, INPUT_DIR, OUTPUT_DIR


def create_local_folders(folder_list):
    folder_list.insert(0, FILE_DIR)
    for folder in folder_list:
        try:
            os.mkdir(folder)
            print(f"Created folder with path: {folder}")
        except FileExistsError:
            print(f"Folder already exists for path: {folder}")


def create_letter_folders(letter_list):
    for folder in letter_list:
        try:
            os.mkdir(f"{INPUT_DIR}/{folder}")
            print(f"Created folder with path: {folder}")
        except FileExistsError:
            print(f"Folder already exists for path: {folder}")


if __name__ == "__main__":
    folders_to_be_created = [FILE_DIR, INPUT_DIR, OUTPUT_DIR]

    letter_list_to_be_created = list(string.ascii_uppercase)

    create_local_folders(folders_to_be_created)
    create_letter_folders(letter_list_to_be_created)
