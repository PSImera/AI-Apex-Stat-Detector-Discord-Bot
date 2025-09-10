# AI Apex Stat Detector <a href="https://discordapp.com/" target="_blank">Discord</a> Bot

> <a href="README-RU.MD" target="_blank">**Это описание есть на русском языке**</a>

The bot is designed to read and parse information - namely statistics - from <a href="images/effect.png">screenshots</a>. It can be configured to give a user either one of two roles (or both) based on the parsed statistics.

![Statistics Channel](images/stat_channel.png)

The 'Skill' role is based on a rating computed from various statistics, while the 'Rank' role is given based on peak rank within the current season. Conceptually, users are supposed to provide screenshots of their statistics in a public channel, after which the bot parses the image, deletes it and informs the user which roles they were given.

![Log Channel](images/log_channel.png)

The bot is also designed to discretely archive sent images in a separate, hidden channel. The intent behind archiving images in a hidden channel is to avoid discouraging users from sending their statistics in the otherwise public 'input' channel.

## Limitations

The bot requires good quality screenshots, otherwise it won't accurately read information. This includes:
- Standard aspect ratio resolutions of 1280x720 or higher
- No heavily compressed or warped (e.g. a photo of a screen or a stretched resolution) images
- Only default (no Reshade, colour-blind setting, etc) colours - important for reading ranks
- There should be nothing blocking the screenshot, for example cursor or popup windows
- English/Russian as supported languages

> Why? See <a href="https://github.com/PSImera/AI-Apex-Stat-Detector-Discord-Bot/blob/main/README.MD#Description-of-the-ML-approach-for-the-bot" target="_blank">here</a>


# 🚀 Installation

1. Clone the repository

``` bash
git clone https://github.com/PSImera/AI-Apex-Stat-Detector-Discord-Bot.git
cd AI-Apex-Stat-Detector-Discord-Bot
```

2. Create a Discord Bot
- Go to <a href="https://discordapp.com/developers/applications/" target="_blank">Discord Developer Portal</a>
- Create an app and get a bot token
- In the root of your project, create a `.env` file and add the line:

```
TOKEN = '<your token>'
```

> Example: <a href=".env.example" target="_blank">.env.example</a>

3. Download Models

- Install Git <a href="https://git-lfs.com/" target="_blank">LFS</a> (if not already installed):

``` bash
git lfs install
```

- Download models from <a href="https://huggingface.co/PSImera/AI-Apex-Stat-Detector-Models" target="_blank">Hugging Face</a> to the `models` folder:

``` bash
mkdir models
cd models
git clone https://huggingface.co/PSImera/AI-Apex-Stat-Detector-Models
git lfs pull
```

4.   for Create virtual environment 

> (for windows users) You can just use `start-bot.bat` for Create virtual environment if it's not created yet and script do it and start the bot. You must have <a href="https://developer.nvidia.com/cuda-toolkit" target="_blank">**CUDA® Toolkit**</a> if you wanna use your nvidia gpu for increase bot performance. Tested on **Python 3.9**

``` bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate.bat     # Windows
pip install --upgrade pip
```

for using without GPU (CPU only):

``` bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

for using with GPU (NVIDIA, CUDA):

``` bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Compiled for CUDA version 12.8
If you have another one, you may need to change torch version
> ⚠️ for experienced users

5. Launch

Return to the root of the project and launch the bot:

``` bash
python main.py
```

# Docker Use case

``` bash
docker pull psimera/apex-stat-detector-bot-gpu
```

You also need to copy the `.env` file with your token to the docker container 

``` bash
docker cp <path_to/.env> <container name>:/bot
```

Copy config file to container if you have it

``` bash
docker cp <path_to/config.json> <container name>:/bot
```

## Supported commands
The bot's settings are saved in `config.json`, which can be edited manually if needed (<a href="config_example.json" target="_blank">example</a>) - but the bot can be fully configured from discord using commands with the prefix `/`

| command | Description |
|---------- |--------------------------------|
| set_channel | Sets the channel for uploading screenshots |
| set_log_channel | Sets the image archive channel |
| set_skill_roles | Used to assign roles for skill, for each individual level (indices from 1 to 6) |
| set_rank_roles | Used to assign roles for ranks (indices from 0 to 7), where 0 corresponds to providing a screenshot without ranked stats, and 1-7 range from bronze to predator |
| set_mode | Used to change operating mode between giving users a skill role, giving users a rank role, or giving both |

## Description of the ML approach for the bot

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

Received information is processed by the <a href="https://lightgbm.readthedocs.io/en/stable/" target="_blank">**LightGBM**</a> model, trained on a dataset of 3294 lines of statistics. The statistics were read from images downloaded from several popular discord servers, where people publicly submit their statistics to get roles. <a href="https://github.com/gageirwin/Discord-Media-Downloader" target="_blank">**Discord Media Downloader**</a> was used for parsing. All images were manually subjectively marked. To mark the images, I created a tool with <a href="https://docs.python.org/3/library/tkinter.html" target="_blank">**tkinter**</a>. The accuracy of the model is:
```
crossval accuracy
0.9461597742887745

test accuracy
0.8203883495145631
```

> Some files used to work on the project, such as markup applications, a notebook for training models, and other small things, are in <a href="https://github.com/PSImera/Tools_for_work_with_datasets" target="_blank">this</a> repository

> In case of questions, ideas or suggestions, feel free to contact me on discord <a href="https://discord.com/users/237271541040021505" target="_blank">**@psimera**</a>