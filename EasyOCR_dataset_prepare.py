import os
import pandas as pd
import cv2
import easyocr
from stat_from_img import read_text_on_image


def crop_text_from_img(input_dir, output_dir):
    reader = easyocr.Reader(['en', 'ru'])
    for file in os.listdir(input_dir):
        name, frmt = file.rsplit('.', 1)
        image = cv2.imread(f'{input_dir}\\{file}')
        results = read_text_on_image(reader, image, detail=1)
        for i, (bbox, text, confidence) in enumerate(results):
            (top_left, top_right, bottom_right, bottom_left) = bbox
            x1, y1 = map(int, top_left)
            x2, y2 = map(int, bottom_right)
            text_block = image[y1:y2, x1:x2]
            output_path = os.path.join(output_dir, f"{name}_{i}.{frmt}")
            cv2.imwrite(output_path, text_block)
        print(f"Все текстовые блоки {file} сохранены")

    print(f"Все изображения обработаны.")


def pre_markup_blocks(input_dir, output_csv):
    reader = easyocr.Reader(
        ['en', 'ru'],
        model_storage_directory='EasyOCR_model',
        user_network_directory='EasyOCR_user_network',
        recog_network='apex_stats_detector'
    )
    results = []
    for i, filename in enumerate(os.listdir(input_dir)):
        img = input_dir + '/' + filename
        result = read_text_on_image(reader, img)

        row = {"file_name": filename}
        row.update({'text': ''.join(result)})
        results.append(row)
        print(f'шаг {i+1}/{len(os.listdir(input_dir))} - {row}')

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f'все блоки размечены, результаты созранены в {output_csv}')


def main():
    input_img_folder = 'datasets\\stats_img'
    blocks_folder = 'datasets\\EasyOCR_train'
    csv_path = f'datasets\\EasyOCR_train\\pre_labels.csv'

    os.makedirs(blocks_folder, exist_ok=True)

    crop_text_from_img(input_img_folder, blocks_folder)
    pre_markup_blocks(blocks_folder, csv_path)
            

if __name__ == "__main__":
    main()