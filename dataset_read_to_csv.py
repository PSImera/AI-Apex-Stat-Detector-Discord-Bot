import os
import pandas as pd
from stat_from_img import stat_by_screen


def main(start_index=0):
    folder = 'datasets\\stat_img'
    output_csv = 'datasets\\dataset_new.csv'
    line_for_clean = ''
    ALL_COLUMNS = [
        "filename", "kills", "deaths", "kd", "counted_kd", "avg", 
        "is_ranked", "season", "rank_games", "rank_kills", "rank_deaths",
        "rank_kd", "rank_counted_kd", "rank_avg", "rank_s1", "rank_s2",
        "season_kills", "season_deaths", "season_kd", 
        "season_counted_kd", "season_avg"
    ]

    if not os.path.exists(output_csv):
        pd.DataFrame(columns=ALL_COLUMNS).to_csv(output_csv, index=False)

    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        processed_files = set(existing_df['filename'])
    else:
        processed_files = set()
    
    files = os.listdir(folder)
    
    for i, file in enumerate(files[start_index:], start=start_index):
        try:
            if file in processed_files:
                continue
            
            line_for_print = f'Processed: {i+1}/{len(files)}'
            line_for_clean = ' ' * len(line_for_print)
            print(line_for_print, end='\r')

            filename = file.rsplit('.', 1)[0]
            path = os.path.join(folder, file)
            
            with open(path, 'rb') as f:
                image = f.read()   
            
            stats = stat_by_screen(image, filename)
            row = {"filename": file}
            row.update(stats)

            for col in ALL_COLUMNS:
                if col not in row:
                    row[col] = 0
            
            pd.DataFrame([row], columns=ALL_COLUMNS).to_csv(output_csv, mode='a', header=False, index=False)
        except Exception as e:
            print(f"\nError on index {i} with file {file}: {e}")
            break
    
    print(line_for_clean, end='\r')
    print(f'process ended, dataset saved as {output_csv}')

if __name__ == "__main__":
    main(start_index=0)
