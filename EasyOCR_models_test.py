import os
import pandas as pd
import easyocr
from PIL import Image
from sklearn.metrics import accuracy_score
from stat_from_img import read_text_on_image


def character_accuracy(y_true, y_pred):
    total_char = 0
    correct_char = 0
    for true, pred in zip(y_true, y_pred):
        total_char += len(true)
        correct_char += sum(1 for a, b in zip(true, pred) if a == b)
    return correct_char / total_char if total_char > 0 else 0


def test_ocr(csv_path, dataset_dir, modelname): 
    reader = easyocr.Reader(['en', 'ru'], 
                    model_storage_directory='EasyOCR_model',
                    user_network_directory='EasyOCR_user_network',
                    recog_network=modelname)
    
    df = pd.read_csv(csv_path)

    results = []
    print(f'\n ---тестируем модель: {modelname}---')
    for i, row in df.iterrows():
        print(f'Обработано: {i+1}/{len(df)}', end='\r')
        filename = row['filename']
        words = row['words']
        img_path = os.path.join(dataset_dir, filename)
        img = Image.open(img_path)
        
        ocr_result = read_text_on_image(reader, img)
        ocr_text = " ".join(ocr_result)
        
        results.append({
            "filename": filename,
            "words": words,
            "predict": ocr_text
        })
    print(' ' * 40, end='\r')
    results_df = pd.DataFrame(results)

    # Оценка метрик
    accuracy = accuracy_score(results_df['words'], results_df['predict'])
    print(f"Accuracy: {accuracy:.4f}")
    
    char_accuracy = character_accuracy(results_df['words'], results_df['predict'])
    print(f"Character Accuracy: {char_accuracy:.4f}")


def main():
    models_dir = 'EasyOCR_model'
    network_dir = 'EasyOCR_user_network'

    for m_file in os.listdir(models_dir):
        modelname, m_frmt = m_file.rsplit('.', 1)
        if m_frmt == 'pth' and modelname != 'craft_mlt_25k':
            for n_file in os.listdir(network_dir):
                path = f'{network_dir}/{n_file}'
                if os.path.isfile(path):
                    n_filename, n_frmt = n_file.rsplit('.', 1)
                    new_path = f"{network_dir}/{modelname}.{n_frmt}"
                    os.rename(path, new_path)
            
            test_ocr('datasets\\EasyOCR_train\\labels.csv', 
                    'datasets\\EasyOCR_train',
                    modelname)
                

if __name__ == "__main__":
    main()


'''
### gthdjt тестирование на разбитой выборке в 1600 изображений

all_data\\train
Accuracy: 0.5933  - Character Accuracy: 0.8308   DEFAULT
Accuracy: 0.8283  - Character Accuracy: 0.8784   MY 1600

all_data\\val
Accuracy: 0.6100  - Character Accuracy: 0.8403   DEFAULT
Accuracy: 0.7750  - Character Accuracy: 0.8794   MY 1600


### тестирование на всей выборке в 10180 изображений

 ---тестируем модель: best_accuracy---
Accuracy: 0.8282
Character Accuracy: 0.8625

 ---тестируем модель: best_norm_ED---
Accuracy: 0.8304
Character Accuracy: 0.8627

 ---тестируем модель: ru_apex_stats_1600---
Accuracy: 0.7866
Character Accuracy: 0.8579


ПО РЕЗУЛЬТАТУ СОХРОНЕНА МОДЕЛЬ best_norm_ED

'''