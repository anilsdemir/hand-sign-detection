from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent

FILE_DIR = Path.joinpath(BASE_DIR, "local")

INPUT_DIR = Path.joinpath(FILE_DIR, "inputs")

OUTPUT_DIR = Path.joinpath(FILE_DIR, "outputs")
