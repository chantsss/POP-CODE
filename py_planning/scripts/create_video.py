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

    # 将图像添加到帧列表中
    for image in images:
        frame = Image.open(os.path.join(image_folder, image))
        frames.append(frame)

    # 保存为 GIF
    if frames != []:
        frames[0].save(video_name, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)
        print('Video saved at ', video_name)
    else:
        print('Empty scenario folder at', image_folder)

    
    


if __name__ == "__main__":
    # image_folders = ["/home/sheng/py_planning/commonroad/exp_plan/hivt5full_ca_1.0_p1/videos", 
    #                  "/home/sheng/py_planning/commonroad/exp_plan/hivt5random_ca_1.0_p1/videos",
    #                  "/home/sheng/py_planning/commonroad/exp_plan/hivt5full_ca+_1.0_p1/videos",
    #                  "/home/sheng/py_planning/commonroad/exp_plan/hivt5random_ca+_1.0_p1/videos"]

    image_folders = [
                    # "/home/sheng/py_planning/commonroad/exp_plan/hivt5random-mtdistillrefine30-skipfar-all_ca_1.0_p1/videos", 
                    #  "/home/sheng/py_planning/commonroad/exp_plan/hivt5random-mtdistillrefine30-skipfar-all_ca+_1.0_p1/videos",
                     "/home/sheng/py_planning/commonroad/exp_plan/POP1084/videos",
                     "/home/sheng/py_planning/commonroad/exp_plan/HiVT1084/videos"]
                    # '/home/sheng/py_planning/commonroad/exp_plan/POP674/videos']


    # image_folders = ["/home/sheng/py_planning/commonroad/exp_plan/hivt5full-all_ca+_1.0_p1/videos", 
    #                  "/home/sheng/py_planning/commonroad/exp_plan/hivt5full_ca+_1.0_p3/videos",
    #                  "/home/sheng/py_planning/commonroad/exp_plan/hivt5full_ca_1.0_p1/videos",

    #                  "/home/sheng/py_planning/commonroad/exp_plan/cv_ca+-all_1.0_p1/videos",
    #                  "/home/sheng/py_planning/commonroad/exp_plan/cv_ca+_1.0_p1/videos",
    #                  "/home/sheng/py_planning/commonroad/exp_plan/cv_ca_1.0_p1/videos",

    #                  "/home/sheng/py_planning/commonroad/exp_plan/hivt5random-mt-distillrefine30-all_ca+_1.0_p1/videos",
    #                  "/home/sheng/py_planning/commonroad/exp_plan/hivt5random-mt-distillrefine30_ca+_1.0_p3/videos",
    #                  "/home/sheng/py_planning/commonroad/exp_plan/hivt5random-mtdistillrefine30_ca_1.0_p1/videos"]


    # 'hivt5full_ca+_1.0_p1',
    # 'hivt5full_ca+_1.0_p3', 
    # 'hivt5full-all_ca+_1.0_p1',
    # 'hivt5full_ca_1.0_p1',

    # 'cv_ca+_1.0_p1',
    # 'cv_ca+-all_1.0_p1',
    # 'cv_ca_1.0_p1',

    # 'hivt5random-mtdistillrefine30_ca+_1.0_p1',
    # 'hivt5random-mt-distillrefine30_ca+_1.0_p3',
    # 'hivt5random-mt-distillrefine30-all_ca+_1.0_p1',
    # 'hivt5random-mtdistillrefine30_ca_1.0_p1'

    duration = 100  # 帧速率
    for image_folder in image_folders:
        for scenario_file in os.listdir(image_folder):
            # scenario_dir = image_folder + '/' + scenario_file
            if 'gif' in scenario_file or 'svg' in scenario_file or 'pdf' in scenario_file:
                continue
            scenario_dir = image_folder + '/' + scenario_file
            video_name = "output_video" + scenario_file + ".gif"
            scenario_dir_video_name = image_folder + '/' + video_name
            images_to_video(scenario_dir, scenario_dir_video_name, duration)