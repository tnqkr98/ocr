import pandas as pd
from file_util import load_files, crop_img

import text_recognition.demo as recognition
import text_detection.test as detection


def run_main():
    detection.run_detection()

    img_files, img_bbox = load_files()
    crop_img(img_files, img_bbox)
    pred_str = recognition.run_recognition()

    # [l, t], [r, t], [r, b], [l, b]
    for i, file in enumerate(img_files):
        txt = pd.read_csv(img_bbox[i], header=None)
        df = pd.DataFrame(columns=["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "result_text"])

        for num, _col in enumerate(list(df)[:-1]):
            df[_col] = txt[num]
        df["result_text"] = pred_str
        df.to_csv("./result.csv")
