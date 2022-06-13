import cv2
import os
import csv


def load_images_from_folder(folder):
    images = []
    for idx, filename in enumerate(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))
        cv2.imshow(str(idx), img)
        key = cv2.waitKey(1)
        print("0-background, 1-sign-20, 2-sign30: \t")
        chosen_idx = input()
        os.system("cls" if os.name == "nt" else "clear")
        if img is not None:
            images.append([filename, chosen_idx])

        path = os.path.join(os.getcwd(), "labels.csv")
        touch_csv_and_save(path, images)

    cv2.destroyAllWindows()


def touch_csv_and_save(path: str, rows: list):
    # open the file in the write mode
    with open(path, "w") as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    load_images_from_folder("imgs_raw")
    # cwd = os.getcwd()
    # print(cwd)
    # touch_csv_and_save(os.path.join(cwd, "labels.csv"), [[1, 2], [3, 4]])
    # print(os.listdir("imgs_raw"))
