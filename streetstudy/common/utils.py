import subprocess

def generate_video(folder):
    """
    Given a folder, generate video from all png
    """
    subprocess.call([
    'ffmpeg', '-framerate', '8', '-pattern_type', 'glob', '-i', f'{folder}/*.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    f'{folder}/output.mp4'
    ])
    return