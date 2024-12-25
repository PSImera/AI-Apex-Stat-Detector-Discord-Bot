import os
import easyocr
import pandas as pd

def main():
    input_dir = "datasets\\easy_OCR_train"
    reader = easyocr.Reader(['ru', 'en'])

    results = []
    for i, file in enumerate(os.listdir(input_dir)):
        path = input_dir + '/' + file
        text = reader.readtext(path, detail=0)

        row = {"filename": file}
        row.update({'words': text})
        results.append(row)
        print(f'шаг {i+1}/{len(os.listdir(input_dir))} - {row}')

    df = pd.DataFrame(results)
    df.to_csv("datasets/text_blocks_markup_for_easy_OCR.csv", index=False)

if __name__ == "__main__":
    main()