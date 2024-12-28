# AI Apex Stat Detector <a href="https://discordapp.com/" target="_blank">Discord</a> Bot

> <a href="README-RU.MD" target="_blank">**Это описание есть на русском языке**</a>

![screenshot example](images/effect.png)

The bot is designed to read and parse information - namely statistics - from screenshots. It can be configured to give a user either one of two roles (or both) based on the parsed statistics.

![Statistics Channel](images/stat_channel.png)

The 'Skill' role is based on a rating computed from various statistics, while the 'Rank' role is given based on peak rank within the current season. Conceptually, users are supposed to provide screenshots of their statistics in a public channel, after which the bot parses the image, deletes it and informs the user which roles they were given.

![Log Channel](images/log_channel.png)

The bot is also designed to discretely archive sent images in a separate, hidden channel. The intent behind archiving images in a hidden channel is to avoid discouraging users from sending their statistics in the otherwise public 'input' channel.

# Usage

- Tested on **Python 3.9** and library versions specified in <a href="requirements.txt" target="_blank">requirements.txt</a>
- Create a bot on <a href="https://discordapp.com/developers/applications/" target="_blank">DiscordDevelopers</a> and copy the token
- Paste your token in the `.env` file in the root folder with the line `TOKEN = '<your token>'` (<a href=".env.example" target="_blank">example</a>)
- To start an instance, run <a href="stat_detector.py" target="_blank">stat_detector.py</a>
- If your machine has an **NVIDIA** gpu, you can run the bot using **CUDA**, which will increase performance. To do this, you must have <a href="https://developer.nvidia.com/cuda-toolkit" target="_blank">**CUDA® Toolkit**</a>, <a href="https://developer.nvidia.com/cudnn" target="_blank">**cuDNN**</a> and a version of <a href="https://pytorch.org/get-started/locally/" target="_blank">**Torch**</a> with GPU support installed (see <a href="requirementsGPU.txt" target="_blank">requirementsGPU.txt</a>)


## Limitations

- To filter out low quality screenshots, the bot will only parse images with a resolution of 1280x720 or higher. This limitation can easily be found in the source code and removed if needed - it is likely to still work on screenshots with lower resolutions.
- The fidelity of provided images directly affects the bot's ability to accurately read information, which means that it will not work well with heavily compressed images or warped images (for example: perspective warping caused by taking an image of the monitor instead of taking a screenshot)
- This includes other forms of warping, most importantly horizontal or vertical stretching
- The bot is sensitive to color and will read information less accurately (most notably ranks) from images with filtered or changed colors
- Provided screenshots must contain the statistics screen in full view, without the cursor and/or popup windows covering anything
- Supported languages include English and Russian

> Why? See <a href="https://github.com/PSImera/AI-Apex-Stat-Detector-Discord-Bot?tab=readme-ov-file#Internals-and-machine-learning" target="_blank">here</a>

## Supported commands
The bot's settings are saved in `config.json`, which can be edited manually if needed (<a href="config_example.json" target="_blank">example</a>) - but the bot can be fully configured from discord using commands with the prefix `/`

| command | Description |
|---------- |--------------------------------|
| set_channel | Sets the channel for uploading screenshots |
| set_log_channel | Sets the image archive channel |
| set_skill_roles | Used to assign roles for skill, for each individual level (indices from 1 to 6) |
| set_rank_roles | Used to assign roles for ranks (indices from 0 to 7), where 0 corresponds to providing a screenshot without ranked stats, and 1-7 range from bronze to predator |
| set_mode | Used to change operating mode between giving users a skill role, giving users a rank role, or giving both |

## Internals and machine learning

The bot works with various different resolutions and supports all standard aspect ratios, which include 16:9, 16:10, 21:9, 4:3, and 5:4. Stretched or warped images are not supported because parsing works by scanning 4 circles for information. If the bot is provided with a screenshot that deviates from its original aspect ratio (e.g. 4:3 stretched to 16:9), the bot is not capable of stretching the circles within which it scans into ellipses to compensate for the warping.
> In the future, I may implement support for stretched screenshots. For the forseeable future, you will have to ask users to provide screenshots where the aspect ratio matches the resolution, in other words - screenshots that aren't stretched. NVIDIA Shadowplay is already known to take screenshots in their original, non-stretched resolutions. Alternatively, you can enforce a standard resolution (and/or aspect ratio) until the functionality is implemented.

Optical character recognition is implemented through use of the **EasyOCR** ​​`cyrillic_g2.pth` model after being retrained on 10180 images with text taken from many different statistics screenshots.
To train the model, <a href="https://github.com/JaidedAI/EasyOCR" target="_blank">**EasyOCR trainer**</a> was used, images were marked up using a homemade text markup application, made using the <a href="https://docs.python.org/3/library/tkinter.html" target="_blank">**tkinter**</a> library. Retraining the model significantly improved its accuracy when reading screenshots.

```
default EasyOCR cyrillic_g2.pth metrics
Accuracy: 0.5933 - Character Accuracy: 0.8308

apex_stats_detector.pth metrics
Accuracy: 0.8304 - Character Accuracy: 0.8627
```
![Damage](images/damage-annot.png)
![KD](images/kd-annot.png)

To differentiate between screenshots showing season statistics and screenshots of ranked season statistics, the bot reads the text within the dropdown menu.

![Title](images/title-annot.png)

The rank is read by using color information from the rank emblems. In some cases, a player may have achieved the same rank in both splits, shown as one emblem instead of two individual splits. In such cases, only one emblem is drawn in the middle and the color information is identical for both splits due to symmetry.

![Split 1](images/split_1.png)
![Split 2](images/split_2.png)

> A player's rank corresponds to their peak rank in the season.

Received information is processed by the <a href="https://lightgbm.readthedocs.io/en/stable/" target="_blank">**LightGBM**</a> model, trained on a dataset of 3300 lines of statistics. The statistics were read from images downloaded from several popular discord servers, where people publicly submit their statistics to get roles. <a href="https://github.com/gageirwin/Discord-Media-Downloader" target="_blank">**Discord Media Downloader**</a> was used for parsing. All images were manually subjectively marked. To mark the images, I created a tool with <a href="https://docs.python.org/3/library/tkinter.html" target="_blank">**tkinter**</a>. The accuracy of the model is:
```
crossval accuracy
0.9461597742887745

test accuracy
0.8203883495145631
```

> If you need to change something in the code, you can set the **DEBUG_PRINT** and/or **DEBUG_IMG** to **True** at the top of <a href="stat_from_img.py" target="_blank">stat_from_img.py</a>, which will (1) output debug information into the terminal and/or (2) save preview images to the specified folder.

> In case of questions, ideas or suggestions, feel free to contact me on discord <a href="https://discord.com/users/237271541040021505" target="_blank">**@psimera**</a>