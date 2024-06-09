import os
import cv2from PIL import Image
import re



def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def images_to_video(image_folder, video_name, duration=100):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    images.sort(key=natural_keys)

    frames = []

    # Load images
    for image in images:
        frame = Image.open(os.path.join(image_folder, image))
        frames.append(frame)

    # Save as GIF
    if frames != []:
        frames[0].save(video_name, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)
        print('Video saved at ', video_name)
    else:
        print('Empty scenario folder at', image_folder)

    
    


if __name__ == "__main__":
    image_folders = [
                     "/home/sheng/py_planning/commonroad/exp_plan/POP1084/videos",
                     "/home/sheng/py_planning/commonroad/exp_plan/HiVT1084/videos"]

    duration = 100  # 100ms per frame
    for image_folder in image_folders:
        for scenario_file in os.listdir(image_folder):
            # scenario_dir = image_folder + '/' + scenario_file
            if 'gif' in scenario_file or 'svg' in scenario_file or 'pdf' in scenario_file:
                continue
            scenario_dir = image_folder + '/' + scenario_file
            video_name = "output_video" + scenario_file + ".gif"
            scenario_dir_video_name = image_folder + '/' + video_name
            images_to_video(scenario_dir, scenario_dir_video_name, duration)