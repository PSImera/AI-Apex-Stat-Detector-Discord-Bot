import os
import shutil
import discord
import json
from dotenv import load_dotenv
from stat_from_img import stat_by_screen
from stat_to_role import role_by_stats
from PIL import Image
import io

load_dotenv()
TOKEN = os.getenv('TOKEN')
CONFIG_PATH = "config.json"

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

config = load_config()


# INITIALISE BOT
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

@client.event
async def on_ready():
    await tree.sync()
    print(f'[BOT] {client.user} is ready')


# CONFIG BLOCK
@tree.command(name="set_channel", description="Set up a channel for upload statistics")
@discord.app_commands.describe(channel="Channel for screenshots uploading")
async def set_channel(interaction: discord.Interaction, channel: discord.TextChannel):
    if interaction.user.guild_permissions.administrator:
        guild_id = str(interaction.guild_id)
        if guild_id not in config:
            config[guild_id] = {}
        config[guild_id]["STAT_CHANNEL_ID"] = channel.id
        save_config(config)
        await interaction.response.send_message(f"Channel {channel.mention} has been successfully configured", ephemeral=True)
    else:
        await interaction.response.send_message("You do not have sufficient permissions to execute this command", ephemeral=True)

@tree.command(name="set_log_channel", description="Set up a channel for logs")
@discord.app_commands.describe(channel="log channel")
async def set_log_channel(interaction: discord.Interaction, channel: discord.TextChannel):
    if interaction.user.guild_permissions.administrator:
        guild_id = str(interaction.guild_id)
        if guild_id not in config:
            config[guild_id] = {}
        config[guild_id]["LOG_CHANNEL_ID"] = channel.id
        save_config(config)
        await interaction.response.send_message(f"Log channel {channel.mention} has been successfully configured", ephemeral=True)
    else:
        await interaction.response.send_message("You do not have sufficient permissions to execute this command", ephemeral=True)

@tree.command(name="set_skill_roles", description="skill roles. From [1] weak to [6] strong")
@discord.app_commands.describe(index="Skill level", role="role on server")
async def set_skill_roles(interaction: discord.Interaction, index: str, role: discord.Role):
    if interaction.user.guild_permissions.administrator:
        guild_id = str(interaction.guild_id)
        if guild_id not in config:
            config[guild_id] = {}
        if "SKILL_ROLES" not in config[guild_id]:
            config[guild_id]["SKILL_ROLES"] = {}
        config[guild_id]["SKILL_ROLES"][index] = role.id
        save_config(config)
        await interaction.response.send_message(f"Role {role.name} for {index} has been successfully configured", ephemeral=True)
    else:
        await interaction.response.send_message("You do not have sufficient permissions to execute this command", ephemeral=True)

@tree.command(name="set_rank_roles", description="rank roles. [0] not ranked & From [1] bronze to [7] predator")
@discord.app_commands.describe(index="Rank index", role="role on server")
async def set_rank_roles(interaction: discord.Interaction, index: str, role: discord.Role):
    if interaction.user.guild_permissions.administrator:
        guild_id = str(interaction.guild_id)
        if guild_id not in config:
            config[guild_id] = {}
        if "RANK_ROLES" not in config[guild_id]:
            config[guild_id]["RANK_ROLES"] = {}
        config[guild_id]["RANK_ROLES"][index] = role.id
        save_config(config)
        await interaction.response.send_message(f"Role {role.name} for {index} has been successfully configured", ephemeral=True)
    else:
        await interaction.response.send_message("You do not have sufficient permissions to execute this command", ephemeral=True)

@tree.command(name="set_mode", description="Switch between three modes")
@discord.app_commands.describe(mode="Choose one of the three modes")
@discord.app_commands.choices(
    mode=[
        discord.app_commands.Choice(name="Skill only", value="skill_only"),
        discord.app_commands.Choice(name="Rank only", value="rank_only"),
        discord.app_commands.Choice(name="Both roles", value="both_roles"),
    ]
)
async def set_mode(interaction: discord.Interaction, mode: discord.app_commands.Choice[str]):
    if interaction.user.guild_permissions.administrator:
        guild_id = str(interaction.guild_id)
        if guild_id not in config:
            config[guild_id] = {}
        config[guild_id]["MODE"] = mode.value
        save_config(config)
        await interaction.response.send_message(f"Bot is now in '{mode.name}' mode.", ephemeral=True)
    else:
        await interaction.response.send_message("You do not have sufficient permissions to execute this command.", ephemeral=True)


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return

    # CHECK SERVER
    guild_id = str(message.guild.id)
    if guild_id not in config:
        return

    # CHECK CHANNEL
    stat_channel_id = config[guild_id].get("STAT_CHANNEL_ID")
    if message.channel.id == stat_channel_id:
        is_stat_channel = True
    else:
        is_stat_channel = False
        return

    # CHECK ATTACHMENTS
    if message.attachments:
        is_attachment = True
    else:
        is_attachment = False
        await message.delete()
        print(f'deleted spam in {message.channel.name} channel by {message.author.name}')
        return

    if len(message.attachments) == 1:
        is_only_one_attachment = True
    else:
        is_only_one_attachment = False
        await message.delete()
        print(f"[error] more then 1 attachment by {message.author.name}")
        await message.channel.send(f'{message.author.mention} [error] send only 1 attachment with your statistic screen please')
        return
    
    attachment = message.attachments[0]
    if attachment.filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        is_image = True
        print(f"[IMG] {attachment.filename} by {message.author.name}")
        bytes_image = await attachment.read()
        bytesio_image = io.BytesIO(bytes_image)
        image = Image.open(bytesio_image)
    else:
        is_image = False
        await message.delete()
        print(f"[error] not image file by {message.author.name}")
        await message.channel.send(f'{message.author.mention} [error] send only image files (png, jpg, jpeg, gif, bmp) with your statistic screen please')
        return
    
    if image.size[0] >= 1280 and image.size[1] >= 720:
        is_resolution_ok = True
    else:
        is_resolution_ok = False
        await message.delete()
        print(f"[error] low resolution image by {message.author.name}")
        await message.channel.send(f'{message.author.mention} [error] send image more then 1280x720')
        return

    # IF ALL OK, READ SCREEN STATS AND GIVE ROLES
    if all([is_stat_channel, is_attachment, is_only_one_attachment, is_image, is_resolution_ok]):
        roles_map_skill = config[guild_id].get("SKILL_ROLES", {})
        roles_map_rank = config[guild_id].get("RANK_ROLES", {})
        log_channel = client.get_channel(config[guild_id].get("LOG_CHANNEL_ID"))
        mode = config[guild_id].get("MODE")

        # save img to temp folder and delete message
        os.makedirs('temp', exist_ok=True)
        file_path = f'temp/{attachment.filename}'
        await attachment.save(file_path)
        await message.delete()

        stats = await stat_by_screen(bytes_image, attachment.filename)
        skill, rank = await role_by_stats(**stats)

        log = f'```python\n' + '\n'.join(f'{key} = {value}' for key, value in stats.items()) + '```'

        # remove current roles from roles list
        current_skill_roles = [rl_name for rl_name, rl_id in roles_map_skill.items() if discord.utils.get(message.guild.roles, id=rl_id) in message.author.roles]
        current_rank_roles = [rl_name for rl_name, rl_id in roles_map_rank.items() if discord.utils.get(message.guild.roles, id=rl_id) in message.author.roles]
        for cr_name in current_skill_roles:
            await message.author.remove_roles(discord.utils.get(message.guild.roles, id=roles_map_skill.get(cr_name)))
        for cr_name in current_rank_roles:
            await message.author.remove_roles(discord.utils.get(message.guild.roles, id=roles_map_rank.get(cr_name)))

        if skill == 'error':
            await message.channel.send(f'{message.author.mention} [error] cant reed stats from this image. try make better screen')
            await log_channel.send(log, file=discord.File(file_path))
            print(f'[BOT] error, {message.author.name} not get role, cant read stats')
            return
        
        if mode == 'both_roles':
            skill_role = discord.utils.get(message.guild.roles, id=roles_map_skill.get(skill))
            rank_role = discord.utils.get(message.guild.roles, id=roles_map_rank.get(rank))
            await message.author.add_roles(skill_role)
            await message.author.add_roles(rank_role)
            await message.channel.send(f'{message.author.mention} you get roles **{skill_role}** and **{rank_role}**')
            print(f'[BOT] {message.author.name} get roles {skill} & {rank}')
            await log_channel.send(f'{message.author.mention}{skill_role.mention}{rank_role.mention}'+log, file=discord.File(file_path))
        elif mode == 'skill_only':
            skill_role = discord.utils.get(message.guild.roles, id=roles_map_skill.get(skill))
            await message.author.add_roles(skill_role)
            await message.channel.send(f'{message.author.mention} you get role **{skill_role}**')
            print(f'[BOT] {message.author.name} get role {skill}')
            await log_channel.send(f'{message.author.mention}{skill_role.mention}'+log, file=discord.File(file_path))
        elif mode == 'rank_only':
            rank_role = discord.utils.get(message.guild.roles, id=roles_map_rank.get(rank))
            await message.author.add_roles(rank_role)
            await message.channel.send(f'{message.author.mention} you get role **{rank_role}**')
            print(f'[BOT] {message.author.name} get role {rank}')
            await log_channel.send(f'{message.author.mention}{rank_role.mention}'+log, file=discord.File(file_path))
        else:
            print(f"something wrong with mode setting")



        os.remove(file_path)
        shutil.rmtree('temp')
        print(f"[BOT] temp files deleted: {file_path}")

    return

if __name__ == '__main__':
    client.run(TOKEN)