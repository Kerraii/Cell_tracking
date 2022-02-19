import subprocess
from pathlib import Path
import os


def parse_result():
    csv_location = Path("src", "cell_detection", "concurrentie", "Cells.csv")
    with open(csv_location, 'r') as csv:
        lines = csv.readlines()
        lines = [line[:-1] for line in lines[1:]]
    previous_id = 0
    current_count = 0
    results = []
    for line in lines:
        image_id = int(line.split(",")[0]) - 1
        if previous_id == image_id:
            current_count += 1
        else:
            results.append(current_count)
            assert(image_id == previous_id + 1)
            previous_id = image_id
            current_count = 1
    results.append(current_count)
    return results


def select_images(input_file):
    input_images = Path("data", "frames", input_file)
    frames = ["{}\\{}\n".format(str(input_images.absolute()), str(f)) for f in os.listdir(input_images)
              if os.path.isfile(Path(input_images, f))]
    target_location = Path("src", "cell_detection", "concurrentie", "images.txt")
    with open(target_location, 'w') as images_txt:
        for frame in frames:
            images_txt.write(frame)


def get_counts(input_file):
    # Execution from project root
    if "concurrentie" in os.getcwd():
        while "src" in os.getcwd():
            os.chdir("..")

    bat_file = Path("src", "cell_detection", "concurrentie", "run.bat").absolute()
    select_images(input_file)
    subprocess.call([bat_file])
    counts = parse_result()
    os.remove(Path("src", "cell_detection", "concurrentie", "Cells.csv"))
    return counts


if __name__ == '__main__':
    video = "A1_mo4_0.75x_2.avi"
    print(get_counts(video))
