import cv2
import numpy as np
from PIL import Image
import re
import asyncio


async def find_circ(image: Image) -> list:
    """FIND CIRCLES ON IMAGE WITH GIVEN RADIUS AS R ARGUMENT"""

    r = image.size[1] / 8.15
    max_deviation = int(image.size[1] * 0.015)

    grey = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    rows = grey.shape[0]
    min_r, max_r = round(r * 0.6), round(r * 1.5)

    circles_cv2 = cv2.HoughCircles(
        grey,
        cv2.HOUGH_GRADIENT,
        1,
        rows / 8,
        param1=200,  # (200) "Canny Edge Detector" threshold, algorithm for finding edges
        # higher = less contours will be found
        param2=80,  # (80) "accumulator Hough Transform" trashold. Count of "votes" to confirm
        # higher = less false circles
        minRadius=min_r,
        maxRadius=max_r,
    )

    circles = (
        [] if circles_cv2 is None else np.uint16(np.around(circles_cv2)).tolist()[0]
    )

    if len(circles) < 4:
        return []

    y_groups = {}
    for circle in circles:
        y = circle[1]
        for base_y in y_groups:
            if abs(base_y - y) <= max_deviation:
                y_groups[base_y].append(circle)
                break
        else:
            y_groups[y] = [circle]

    valid_groups = [(y, group) for y, group in y_groups.items() if len(group) >= 4]
    if not valid_groups:
        return []

    best_group = max(valid_groups, key=lambda x: len(x[1]))
    four_circles = sorted(best_group[1], key=lambda c: c[0])[-4:]

    return four_circles


async def read_text_on_image(reader, image: Image) -> list:
    """READ TEXT FROM IMAGE EITH THE SETTINGS THAT WERE FOUND FOR THIS CASE"""

    return reader.readtext(
        np.array(image),
        min_size=10,  # (10) block size in pixels. If smaller, will be ignored. Increase if there is a lot of noise
        low_text=0.2,  # (0.4) Threshold of "light" text (with low pixel intensity). Helps to highlight low-contrast elements
        mag_ratio=4,  # (1) Image scaling coefficient before processing. Allows to improve recognition of small details
        text_threshold=0.9,  # (0.7) Confidence threshold that this is text. Higher value = less noise
        link_threshold=0.2,  # (0.4) Proximity of blocks for connection. Reducing will combine more
        width_ths=1.0,  # (0.5) Threshold of text block width. Increasing will combine more
        x_ths=0.3,  # (1.0) Proximity threshold of text location to each other by X for combining. Decrease will merge more
        y_ths=0.5,  # (0.5) same in Y
        height_ths=0.5,  # (0.5) Similarity of block heights to merge. Increase to merge more
        ycenter_ths=0.5,  # (0.5) Similarity of block centers in Y to merge. Share of first block height. Increase to merge more
    )


async def ranked_check(reader, image: Image) -> tuple:
    """READ RIGHT BLOCK TOP TEXT"""

    results = await read_text_on_image(reader, image)
    result = [text for _, text, _ in results]
    text = "".join(result)

    if any(key in text.lower() for key in ("rank", "рейт")):
        is_ranked = True
        season = "".join([c for c in text if c.isdigit()])
        if season:
            season = int(season)

    else:
        is_ranked = False
        season = None

    return (is_ranked, season)


async def rank_games_data_processing(reader, image: Image) -> int:
    results = await read_text_on_image(reader, image)
    text_list = [text for _, text, _ in results]

    rg_str = text_list[-1].replace(",", ".")
    rg_str = re.sub(r'"|[A-Za-zА-Яа-я]\.', "", rg_str)
    if rg_str.lower().endswith(("k", "т")):
        rg_str = "".join([c for c in rg_str if c.isdigit() or c in "."])
        rg_float = float(rg_str) * 1000 if rg_str else 0
    else:
        rg_str = "".join([c for c in rg_str if c.isdigit() or c in "."])
        rg_float = float(rg_str) if rg_str else 0

    return int(rg_float) if rg_float else 0


async def rank_detect(image: Image) -> tuple:
    """DETECT RANK BY MEDIAN HUE OF EMBLEM"""

    COLORS_MAP = {
        "predator": ((0, 137.5, 80), (16.5, 255, 255)),  # median H=12
        "master": ((130, 40, 80), (140, 255, 255)),  # median H=135
        "diamond": ((100, 40, 80), (105, 255, 255)),  # median H=103
        "platinum": ((88, 40, 80), (93, 180, 180)),  # median H=90
        "gold": ((16.5, 40, 80), (22, 255, 140)),  # median H=18
        "silver": ((0, 0, 77), (180, 30, 150)),  #
        "bronze": (
            (10, 40, 80),
            (15, 137.5, 150),
        ),  # [ 12. 124. 111.]   rookie[14. 93. 96.]
    }

    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    mask_value = hsv[:, :, 2] > 65  #  V threshold
    mask = mask_value
    if not mask.any():
        return "unknown"
    filtered_hsv = hsv[mask]
    mean_hsv = np.median(filtered_hsv, axis=0)
    for rank, (lower, upper) in COLORS_MAP.items():
        if all(lower < mean_hsv) and all(mean_hsv <= upper):
            return rank

    return "unknown"


async def circles_data_processing(crc_1: list, crc_2: list) -> tuple:
    """GET CIRCLES DATA
    1st argument: list of text from Damage block
    2nd argument: list of text from Kills/Deats block
    """
    kills, deaths, kd, counted_kd, avg = None, None, None, None, None

    if len(crc_1) >= 3:
        avg_str = crc_1[-1].replace(",", ".")
        avg_str = re.sub(r'"|[A-Za-zА-Яа-я]\.', "", avg_str)
        avg_str = "".join(i for i in avg_str.rsplit(".", 1)[0] if i.isdigit())
        avg = int(avg_str) if avg_str else 0

    if len(crc_2) >= 3:
        kills_str = crc_2[0].replace('"', "")
        if kills_str.endswith((".00", ",00", ".0", ",0")):
            kills_str = kills_str[:-2]
        kills_str = "".join(i for i in kills_str if i.isdigit())
        kills = int(kills_str) if kills_str else 0

        deaths_str = crc_2[1].replace('"', "")
        if deaths_str.endswith((".00", ",00", ".0", ",0")):
            deaths_str = deaths_str[:-2]
        deaths_str = "".join(i for i in deaths_str if i.isdigit())
        deaths = int(deaths_str) if deaths_str else 0

        kd_str = crc_2[-1].replace('"', "")
        kd_str = kd_str.replace(",", ".")
        kd_str = "".join(i for i in kd_str if i.isdigit() or i in ".")
        kd = float(kd_str) if kd_str else 0

        counted_kd = kills if not deaths else round(kills / deaths, 2)

    return (kills, deaths, kd, counted_kd, avg)


async def stat_by_screen(image: Image, reader, filename: str) -> dict:
    """MAIN FUNCTION TO COMBINE ALL RPROCESES OF GETING INFO FROM IMAGE

    coordinates based on proportions between circelse with base data

    step 1: found circles
    step 2: read text in circled
    step 3: save global block in dictionary
    step 4: read title of block 2
    IF its ranked:
        step 5.1: read season in title and save in dictionary
        step 5.2: read count of ranked games and save in dictionary
        step 5.3: save right block in dictionary as ranked stats
        step 5.4: detect ranked by split emblems and save in dictionary
    ELSE:
        step 5.5: save right block in dictionary as season stats
    step 6: return result
    """

    res = {}

    # STEP 1
    four_circles = await find_circ(image)

    if four_circles:
        coor_list = []  # list of coordinates for image crop
        CC = []  # Circles Centers
        for circ in four_circles:
            cX, cY, cR = circ[0], circ[1], circ[2]
            CC.append((cX, cY))
            coor_list.append((cX - cR, cY - cR * 0.3, cX + cR, cY + cR))

        # STEP 2
        tasks = [read_text_on_image(reader, image.crop(coor)) for coor in coor_list]
        results = await asyncio.gather(*tasks)

        circ_texts = []
        for result in results:
            text = [text for _, text, _ in sorted(result, key=lambda x: x[0][0][1])]
            circ_texts.append(text)

        # STEP 3
        res["kills"], res["deaths"], res["kd"], res["counted_kd"], res["avg"] = (
            await circles_data_processing(circ_texts[0], circ_texts[1])
        )

        # STEP 4
        D = int(
            (CC[1][0] - CC[0][0] + CC[3][0] - CC[2][0]) / 2
        )  # distance between 2 circles
        CYL = int((CC[0][1] + CC[1][1] + CC[2][1] + CC[3][1]) / 4)  # Circles mean Y
        x_left = CC[2][0] - (D / 3)
        x_center = (CC[2][0] + CC[3][0]) / 2
        x_right = CC[3][0] + (D / 3)
        t_top = CYL - (D * 1.66)
        t_bottom = CYL - (D * 1.41)

        title_coor = (x_left, t_top, x_right, t_bottom)
        title_crop = image.crop(title_coor)
        title_res = await read_text_on_image(reader, title_crop)
        right_block_list = [text for _, text, _ in title_res]
        right_block_title = "".join(right_block_list)
        is_ranked = False
        if any(key in right_block_title.lower() for key in ("rank", "рейт")):
            is_ranked = True
        res["is_ranked"] = is_ranked

        if is_ranked:
            # STEP 5.1
            res["season"] = "".join([c for c in right_block_title if c.isdigit()])

            # STEP 5.2
            rank_games_coor = (
                CC[2][0] - (D * 0.45),
                CYL - (D * 1.41),
                CC[2][0] + (D * 0.1),
                CYL - D,
            )
            res["rank_games"] = await rank_games_data_processing(
                reader, image.crop(rank_games_coor)
            )

            # STEP 5.3
            (
                res["rank_kills"],
                res["rank_deaths"],
                res["rank_kd"],
                res["rank_counted_kd"],
                res["rank_avg"],
            ) = await circles_data_processing(circ_texts[2], circ_texts[3])

            # STEP 5.4
            splits_coor_list = [
                (x_left, CYL - D, x_center, CYL - (D / 2)),
                (x_center, CYL - D, x_right, CYL - (D / 2)),
            ]
            split_tasks = [rank_detect(image.crop(coor)) for coor in splits_coor_list]
            splits = await asyncio.gather(*split_tasks)
            res["rank_s1"] = splits[0]
            res["rank_s2"] = splits[1]
        else:
            # STEP 5.5
            (
                res["season_kills"],
                res["season_deaths"],
                res["season_kd"],
                res["season_counted_kd"],
                res["season_avg"],
            ) = await circles_data_processing(circ_texts[2], circ_texts[3])

    else:
        print(f"Error! Can't find circles correctly in {filename}")

    # STEP 6
    return res
