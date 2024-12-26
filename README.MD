# AI Apex Stat Detector <a href="https://discordapp.com/" target="_blank">Discord</a> Bot

> <a href="README-RU.MD" target="_blank">**Это описание есть на русском языке**</a>

<img src="images/effect.png" width="560" height="302"/>

This bot get screenshot of statistics, reads the information from it and, based on these statistics, gives the user a role or two roles depending on mode.

![Statistics Channel](images/stat_channel.png)

The Skill role based on skill and the Rank role based on maximum rank at season.
Messages with screenshots must be sent in special channel. The bot immediately deletes this message and replies to the user about the roles given to him.

![Log Channel](images/log_channel.png)

The bot saves the sent image into log channel with statistic information and roles given. It is assumed that the log channel is visible only for admins so that people do not hesitate to dump their statistics in a public channel, but you can save this log in the same channel.

# Usage

- Tested on **Python 3.9** and library versions specified in <a href="requirements.txt" target="_blank">requirements.txt</a>
- Make bot on <a href="https://discordapp.com/developers/applications/" target="_blank">DiscordDevelopers</a> and cory token
- Paste your token in the `.env` file in the root folder with the line `TOKEN = '<your token>'` (<a href=".env.example" target="_blank">example</a>)
- For start, run `stat_detector.py`
- If your machine has an **NVIDIA** gpu, you can run bot using **CUDA**, which will increase the speed. To do this, you must have <a href="https://developer.nvidia.com/cuda-toolkit" target="_blank">**CUDA® Toolkit**</a>, <a href="https://developer.nvidia.com/cudnn" target="_blank">**cuDNN**</a> and <a href="https://pytorch.org/get-started/locally/" target="_blank">**Torch**</a> version with GPU support installed (present in <a href="requirementsGPU.txt" target="_blank">requirementsGPU.txt</a>)


## Limitations

- Bot works with a resolution of 1280x720 and higher (but if you want to remove this limitation, it is easy to find in the code. Most likely, the bot will find information on smaller images, but I added this limitation to filter out poor quality screenshots)
- Image quality affects the quality of reading information, so the quality must be good (not a photo of the screen, for example)
- Bot doesn't work well with stretched resolution screenshots. Most likely it will give an error
- Bot doesn't perceive information well (mainly rank) from images with distorted colors
- It is necessary that during the screenshot all information is visible and the mouse hovered over some area does not display a pop-up window that overlaps the information
- The game language must be english or russian

> Why it like this? Read **How it works** section *

## slash commands
The settings saved in `config.json` file, and you can edit it manually if anything (<a href="config_example.json" target="_blank">example</a>), but all the settings are made using slash commands

| command | Description |
|---------- |--------------------------------|
| set_channel | select the channel for screenshot uploading |
| set_log_channel | select the channel for saving logs |
| set_skill_roles | select the list of roles assigned by skill level (indices from 1 to 6) |
| set_rank_roles | select the list of roles assigned by rank (indices from 0 to 7) where 0 for screen without ranked stats, and 1-7 are ranks from bronze to predator |
| set_mode | select from 3 operating modes (both roles or special one) |

## Details of the bot and model training

The bot works with different resolutions, it can find images with either wide screen or 4:3 or 5:4 images, but it should not be stretched because detecting based on search for 4 circles with information and if bot doesn't find them, it returns an error, since elipses are obviously not circles.
> Perhaps in the next versions I will add the functionality of working with stretched images, but for now you will have to ask users with a non-standard resolution to make screenshots not stretched (for example, GeForce makes a screenshot with the set resolution) or set standart resolution temporarily for get screenshot.

Text reading use **EasyOCR** ​​`cyrillic_g2.pth` model retrained on 10180 images with text taken from different statistics screenshots.
For training, <a href="https://github.com/JaidedAI/EasyOCR" target="_blank">**EasyOCR trainer**</a> was used, images were marked up using a homemade text markup application made using the <a href="https://docs.python.org/3/library/tkinter.html" target="_blank">**tkinter**</a> library. After training, the accuracy of reading the font from statistics screenshots has increased significantly

```
default EasyOCR cyrillic_g2.pth metrics
Accuracy: 0.5933 - Character Accuracy: 0.8308

apex_stats_detector.pth metrics
Accuracy: 0.8304 - Character Accuracy: 0.8627
```
![Damage](images/damage-annot.png)
![KD](images/kd-annot.png)

To determine whether a specific screenshot is ranked or global statistics, bot use title of right block

![Title](images/title-annot.png)

To determine the rank, bot use color information from the split emblem. In the case of a season with one split, the emblem determines both splits.

![Split 1](images/split_1.png)
![Split 2](images/split_2.png)

> the rank is given by the maximum rank in season

Then all the received information is processed by the <a href="https://lightgbm.readthedocs.io/en/stable/" target="_blank">**LightGBM**</a> model trained on a dataset of 3300 lines of statistics. The statistics were read from images downloaded from several popular discord servers, where people publicly publish their statistics to get a role. <a href="https://github.com/gageirwin/Discord-Media-Downloader" target="_blank">**Discord Media Downloader**</a> was used for parsing. All images were manually subjectively marked. For marking i again use self made application on <a href="https://docs.python.org/3/library/tkinter.html" target="_blank">**tkinter**</a>. The accuracy of the model is:
```
crossval accuracy
0.9461597742887745

test accuracy
0.8203883495145631
```

> If you need to change something in code, u can turn on by changing to True variables **DEBUG_PRINT** and/or **DEBUG_IMG** on top of the file <a href="stat_from_img.py" target="_blank">stat_from_img.py</a>, which will allow you to output individual details to the terminal and/or save preview images to the specified folder

> If you have any questions or ideas, you can contact me on discord **@psimera**