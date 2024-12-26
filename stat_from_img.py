import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
import io
import re
import os
from itertools import combinations

DEBUG_PRINT = False
DEBUG_IMG = False
debug_folder = 'image_parts'

if DEBUG_IMG:
    os.makedirs(debug_folder, exist_ok=True)


async def find_circ(image: Image, r: int) -> list:
    '''FIND CIRCLES ON IMAGE WITH GIVEN RADIUS AS R ARGUMENT'''

    grey = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    rows = grey.shape[0]
    min_r, max_r = round(r * 0.6), round(r * 1.5) # (0.8 1.2) permissible range of radius deviation
                                                  # if not croped to hd, need more range becouse radius based on image height
                                                  # for example radius of 1080p is 132.5, but circles on 1440x1080 image is 100 pixels
                                                  # its possible to use 0.75 min_r multiplier, 
                                                  # but i crop image on bottom size in this code so it should work with not big ranges

    circles_cv2 = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 1, rows / 8,
        param1=200, # (200) "Canny Edge Detector" threshold, algorithm for finding edges
                    # higher = less contours will be found
        param2=80,  # (80) "accumulator Hough Transform" trashold. Count of "votes" to confirm
                    # higher = less false circles
        minRadius=min_r, maxRadius=max_r)

    return [] if circles_cv2 is None else np.uint16(np.around(circles_cv2)).tolist()[0]


async def four_circles_in_line(circles: list, max_deviation: int) -> list:
    '''FILTER ONLY 4 RIGHT CIRCLES WITH ALMOST SAME Y COORDINATE'''

    valid_combinations = []
    for subset in combinations(circles, 4):
        ys = [circle[1] for circle in subset] 
        if max(ys) - min(ys) <= max_deviation:
            valid_combinations.append(sorted(subset, key=lambda c: c[0]))
    if not valid_combinations:
        return []
    return sorted(valid_combinations, key=lambda c: c[0][0])[-1]


async def read_text_on_image(reader, image: Image) -> list:
    '''READ TEXT FROM IMAGE EITH THE SETTINGS THAT WERE FOUND FOR THIS CASE'''

    return reader.readtext(np.array(image),
                            min_size=10, # (10) block size in pixels. If smaller, will be ignored. Increase if there is a lot of noise
                            low_text=0.2, # (0.4) Threshold of "light" text (with low pixel intensity). Helps to highlight low-contrast elements
                            mag_ratio=4, # (1) Image scaling coefficient before processing. Allows to improve recognition of small details
                            text_threshold=0.9, # (0.7) Confidence threshold that this is text. Higher value = less noise
                            link_threshold=0.2, # (0.4) Proximity of blocks for connection. Reducing will combine more
                            width_ths=1.0, # (0.5) Threshold of text block width. Increasing will combine more
                            x_ths=0.3, # (1.0) Proximity threshold of text location to each other by X for combining. Decrease will merge more
                            y_ths=0.5, # (0.5) same in Y
                            height_ths=0.5, # (0.5) Similarity of block heights to merge. Increase to merge more
                            ycenter_ths=0.5, # (0.5) Similarity of block centers in Y to merge. Share of first block height. Increase to merge more
                            )

async def draw_readed_text(image: Image, ocr_results, filename: str):
    '''FUNCTION FOR DEBUG ONLY, SAVE IMAGE WITH DRAWED TEXT ON IT'''

    font_path = 'TTSquaresCondensed-Regular.ttf'  # cyrylic font
    image_pil = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))  # convert for PIL
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, 20)  # 2nd arg is size in pixels
    
    # draw blocks
    for (bbox, text, confidence) in ocr_results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        draw.rectangle([top_left, bottom_right], outline=(0, 255, 0), width=2)
        text_position = (top_left[0], top_left[1] - 15)

        # draw text with outline
        x, y = text_position
        text_color = (255, 0, 0)
        outline_color = (0, 0, 0)
        outline_width = 2
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, fill=outline_color, font=font)
        draw.text((x, y), text, fill=text_color, font=font)

    # convert to OpenCV and save image
    image_annotated = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{debug_folder}\\{filename}-annot.png', image_annotated)
    return


async def ranked_check(reader, image: Image, filename: str) -> tuple:
    '''READ RIGHT BLOCK TOP TEXT'''

    results = read_text_on_image(reader, image)
    if DEBUG_IMG:
        await draw_readed_text(image, results, filename)
    result = [text for _, text, _ in results]
    text = ''.join(result)

    if any(key in text.lower() for key in ('rank', 'рейт')):
        is_ranked = True
        season = ''.join([c for c in text if c.isdigit()])
        if season:
            season = int(season)

    else:
        is_ranked = False
        season = None

    if DEBUG_PRINT:
        print(f'readed text = {text}')
        print(f'ranked = {is_ranked}, season = {season}')
    
    return (is_ranked, season)


async def rank_detect(image: Image, filename: str) -> tuple:
    '''DETECT RANK BY MEDIAN HUE OF EMBLEM'''

    COLORS_MAP = {
        'predator': ((0, 137.5, 80), (16.5, 255, 255)), # median H=12
        'master': ((130, 40, 80), (140, 255, 255)), # median H=135
        'diamond': ((100, 40, 80), (105, 255, 255)), # median H=103
        'platinum': ((88, 40, 80), (93, 180, 180)), # median H=90
        'gold': ((16.5, 40, 80), (22, 255, 140)), # median H=18
        'silver': ((0, 0, 77), (180, 30, 150)), # 
        'bronze': ((10, 40, 80), (15, 137.5, 150)) # [ 12. 124. 111.]   rookie[14. 93. 96.]
        }

    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    mask_value = hsv[:, :, 2] > 65 #  V threshold
    mask = mask_value
    if not mask.any():
        if DEBUG_PRINT:
            print('hsv mask empty')
        return "unknown"
    filtered_hsv = hsv[mask]

    mean_hsv = np.median(filtered_hsv, axis=0)
    
    # detect rank
    for rank, (lower, upper) in COLORS_MAP.items():
        if DEBUG_PRINT:
            print()
            print(rank)
            print(lower, 'lower')
            print(mean_hsv, 'mean hsv')
            print(upper, 'upper')
            print(all(lower < mean_hsv) and all(mean_hsv <= upper))

        if all(lower < mean_hsv) and all(mean_hsv <= upper):
            # saving image with green mask
            if DEBUG_IMG: 
                visualization = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
                visualization[~mask] = (0, 255, 0)
                cv2.imwrite(f'{debug_folder}\\{filename}_mask_{rank}.png', visualization)
            return rank

    return "unknown"


async def circles_data_processing(crc_1: list, crc_2: list, filename: str) -> tuple:
    '''GET CIRCLES DATA
    1st argument: list of text from Damage block
    2nd argument: list of text from Kills/Deats block
    '''
    kills, deaths, kd, counted_kd, avg = None, None, None, None, None

    if len(crc_1) >= 3:
        avg_str = crc_1[-1].replace(',', '.')
        avg_str = re.sub(r'"|[A-Za-zА-Яа-я]\.', '', avg_str)
        avg_str = ''.join(i for i in avg_str.rsplit('.', 1)[0] if i.isdigit())
        if avg_str:
            avg = int(avg_str)
    else:
        print(f"Error! Can't read Damage info in {filename}")

    if len(crc_2) >= 3:
        kills_str = crc_2[0].replace('"', '')
        if kills_str.endswith(('.00', ',00', '.0', ',0')):
            kills_str = kills_str[:-2]
        kills_str = ''.join(i for i in kills_str if i.isdigit())
        if kills_str:
            kills = int(kills_str) # need fix case of bad screen
        
        deaths_str = crc_2[1].replace('"', '')
        if deaths_str.endswith(('.00', ',00', '.0', ',0')):
            deaths_str = deaths_str[:-2]
        deaths_str = ''.join(i for i in deaths_str if i.isdigit())
        if deaths_str:
            deaths = int(deaths_str) # need fix case of bad screen

        kd_str = crc_2[2].replace('"', '')
        kd_str = kd_str.replace(',', '.')
        kd_str = ''.join(i for i in kd_str if i.isdigit() or i in '.')
        if kd_str:
            kd = float(kd_str)

        if not deaths:
            counted_kd = kills
        else:
            counted_kd = round(kills / deaths, 2)
    else:
        print(f"Error! Can't read KD info in {filename}")
    
    return (kills, deaths, kd, counted_kd, avg) 


async def stat_by_screen(img: bytes, filename: str) -> dict: 
    '''MAIN FUNCTION TO COMBINE ALL RPROCESES OF GETING INFO FROM IMAGE

    coordinates based on proportions between circelse with base data
    
    step 0: prepare to process
    step 1: found circles
    step 2: read text in circled
    step 3: save global block in dictionary
    step 4: read title of block 2
    IF its ranked:
        step 4.1: read season in title and save in dictionary
        step 4.2: detect ranked by split emblems and save in dictionary
        step 4.3: save right block in dictionary as ranked stats
    ELSE:
        step 4.1: save right block in dictionary as season stats
    step 4: return result
    '''

    # STEP 0
    image = Image.open(io.BytesIO(img))
    R = image.size[1] / 8.15
    res = {}

    
    # STEP 1
    circles = sorted(await find_circ(image, R), key=lambda x: x[0])
    if DEBUG_PRINT:
        print('Circles list', circles)
    circles = await four_circles_in_line(circles, int(image.size[1]*0.015))

    if len(circles) == 4:
        reader = easyocr.Reader(['en', 'ru'], 
                    model_storage_directory='EasyOCR_model',
                    user_network_directory='EasyOCR_user_network',
                    recog_network='apex_stats_detector')
        
        coor_list = [] # list of coordinates for image crop 
        CC = [] # Circles Centers
        for circ in circles:
            cX, cY, cR = circ[0], circ[1], circ[2]
            CC.append((cX, cY))
            coor_list.append((cX-cR, cY-cR*0.3, cX+cR, cY+cR))

        
        # STEP 2
        glob_dmg_crop = image.crop(coor_list[0])
        glob_kd_crop = image.crop(coor_list[1])
        right_dmg_crop = image.crop(coor_list[2])
        right_kd_crop = image.crop(coor_list[3])

        glob_dmg_results = await read_text_on_image(reader, glob_dmg_crop)
        glob_kd_results = await read_text_on_image(reader, glob_kd_crop)
        right_dmg_results = await read_text_on_image(reader, right_dmg_crop)
        right_kd_results = await read_text_on_image(reader, right_kd_crop)

        if DEBUG_IMG:
            await draw_readed_text(glob_dmg_crop, glob_dmg_results, filename+'_global_damage')
            await draw_readed_text(glob_kd_crop, glob_kd_results, filename+'_global_kd')
            await draw_readed_text(right_dmg_crop, right_dmg_results, filename+'_right_damage')
            await draw_readed_text(right_kd_crop, right_kd_results, filename+'_right_kd')

        glob_dmg_text = [text for _, text, _ in sorted(glob_dmg_results, key=lambda x: x[0][0][1])]
        glob_kil_text = [text for _, text, _ in sorted(glob_kd_results, key=lambda x: x[0][0][1])]
        right_dmg_text = [text for _, text, _ in sorted(right_dmg_results, key=lambda x: x[0][0][1])]
        right_kil_text = [text for _, text, _ in sorted(right_kd_results, key=lambda x: x[0][0][1])]
        
        if DEBUG_PRINT:
            print(f'{filename} Global Damage block:', glob_dmg_text)
            print(f'{filename} Global Kills block:', glob_kil_text)
            print(f'{filename} Right Damage block:', right_dmg_text)
            print(f'{filename} Right Kills block:', right_kil_text)

        res['kills'], res['deaths'], res['kd'], res['counted_kd'], res['avg']  = await circles_data_processing(
            glob_dmg_text, glob_kil_text, filename+'_global')

        
        # STEP 3
        D = int((CC[1][0] - CC[0][0] + CC[3][0] - CC[2][0]) / 2) # distance between 2 circles
        CYL = int((CC[0][1] + CC[1][1] + CC[2][1] + CC[3][1]) / 4) # Circles mean Y
        x_left = CC[2][0]-(D/3)
        x_center = (CC[2][0] + CC[3][0]) / 2
        x_right = CC[3][0]+(D/3)
        t_top = CYL - (D * 1.66)
        t_bottom = CYL - (D * 1.41)
    
        title_coor = (x_left, t_top, x_right, t_bottom)
        title_crop = image.crop(title_coor)
        title_res = await read_text_on_image(reader, title_crop)
        if DEBUG_IMG:
            await draw_readed_text(title_crop, title_res, filename)
        right_block_list = [text for _, text, _ in title_res]
        right_block_title = ''.join(right_block_list)
        if DEBUG_PRINT:
            print(f'right_block_title = {right_block_title}')
        is_ranked = False
        if any(key in right_block_title.lower() for key in ('rank', 'рейт')):
            is_ranked = True
        res['is_ranked'] = is_ranked       

        
        # STEP 3.1
        if is_ranked:
            res['season'] = ''.join([c for c in right_block_title if c.isdigit()])

            g_left = CC[2][0]-(D*0.45)
            g_top = CYL - (D * 1.41)
            g_right = CC[2][0]+(D*0.1)
            g_bottom = CYL - D
            rank_games_coor = (g_left, g_top, g_right, g_bottom)
            rank_games_crop = image.crop(rank_games_coor)
            rank_games_results = await read_text_on_image(reader, rank_games_crop)

            if DEBUG_IMG:
                await draw_readed_text(rank_games_crop, rank_games_results, filename+'_rank_games')

            rank_games_list = [text for _, text, _ in rank_games_results]

            if DEBUG_PRINT:
                print('ramk_games text', rank_games_list)

            rg_str = rank_games_list[-1].replace(',', '.')
            rg_str = re.sub(r'"|[A-Za-zА-Яа-я]\.', '', rg_str)
            if rg_str.lower().endswith(('k', 'т')):
                rg_str = ''.join([c for c in rg_str if c.isdigit() or c in '.'])
                rg_float = float(rg_str) * 1000 if rg_str else 0
            else:
                rg_str = ''.join([c for c in rg_str if c.isdigit() or c in '.'])
                rg_float = float(rg_str) if rg_str else 0
        
            res['rank_games'] = int(rg_float) if rg_float else 0
            res['rank_kills'], res['rank_deaths'], res['rank_kd'], \
                res['rank_counted_kd'], res['rank_avg'] = await circles_data_processing(
                    right_dmg_text, right_kil_text, filename + '_ranked')
            
            # STEP 3.2
            s_top = CYL - D
            s_bottom = CYL - (D/2)
            split_1_coor = (x_left, s_top, x_center, s_bottom)
            split_2_coor = (x_center, s_top, x_right, s_bottom)
            split_1_crop = image.crop(split_1_coor)
            split_2_crop = image.crop(split_2_coor)

            split_1 = await rank_detect(split_1_crop, filename+'split 1')
            split_2 = await rank_detect(split_2_crop, filename+'split 2')

            if DEBUG_PRINT: 
                print('split_1 =', split_1)
                print('split_2 =', split_2)

            res['rank_s1'] = split_1
            res['rank_s2'] = split_2
        else:
            res['season_kills'], res['season_deaths'], res['season_kd'], \
                res['season_counted_kd'], res['season_avg']  = await circles_data_processing(
                    right_dmg_text, right_kil_text, filename+'_season')

    else:
        print(f'Error! Cannot find circles correctly in {filename}')

    return res