# %%

#-------- import the base packages -------------
import sys
import os
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F
from base64 import b64encode
import numpy as np
from PIL import Image
import cv2
import argparse
from moviepy.editor import ImageSequenceClip
import torchvision.transforms as transforms

import sys

# #-------- import cotracker -------------
# from SpaTracker.models.cotracker.utils.visualizer import Visualizer, read_video_from_path
# from SpaTracker.models.cotracker.predictor import CoTrackerPredictor

# #-------- import spatialtracker -------------
# from SpaTracker.models.spatracker.predictor import SpaTrackerPredictor
# from SpaTracker.models.spatracker.utils.visualizer import Visualizer, read_video_from_path


class SpaTrackerWrapper:
    def __init__(self,
                 root_dir='./',
                 video_name='video',
                 gpu=0,
                 model='cotracker',
                 crop=False,
                 crop_factor=1,
                 debug=False,
                 ):
        
        self.root_dir = root_dir
        self.video_name = video_name
        self.gpu = gpu
        self.model = model
        self.crop = crop
        self.crop_factor = crop_factor
        self.debug = debug

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)

    def mkv_to_frames(self, video_path: str) -> np.ndarray:
        """
        Converts a MKV video file to a sequence of frames.
        This function reads a video file from the specified path, extracts each frame,
        and returns the frames as a numpy array.
        Args:
            video_path (str): The file path to the MKV video.
        Returns:
            np.ndarray: An array containing the frames with shape 
                        (num_frames, height, width, channels), where num_frames is the 
                        number of frames in the video, and height, width, and channels are 
                        the dimensions of each frame.
        """

        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)

        cap.release()
        frames = np.array(frames)
        
        return frames

    def mkv_to_depth_frames(self, video_path: str) -> np.ndarray:
        """
        Converts a MKV video file to a sequence of depth frames.
        This function reads a video file from the specified path, extracts each frame,
        converts it to a depth frame (assuming the depth information is stored in a 
        16-bit single channel format), and returns the depth frames as a numpy array.
        Args:
            video_path (str): The file path to the MKV video.
        Returns:
            np.ndarray: An array containing the depth frames with shape 
                        (num_frames, height, width), where num_frames is the 
                        number of frames in the video, and height and width are 
                        the dimensions of each frame.
        """

        cap = cv2.VideoCapture(video_path)
        depth_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Assuming the depth information is stored in the 16-bit single channel format
            depth_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            depth_frames.append(depth_frame)

        cap.release()
        depth_frames = np.array(depth_frames)

        if self.debug:
            output_dir = os.path.join(self.root_dir, self.video_name)
            os.makedirs(output_dir, exist_ok=True)
            for i, depth_frame in enumerate(depth_frames):
                output_path = os.path.join(output_dir, f'depth_frame_{i+1}.npy')
                np.save(output_path, depth_frame)
        
        return depth_frames
    
    def load_mask(self, image_path):
        """
        Loads a binary mask from a PNG image file.
        This function reads a PNG image file from the specified path, converts it to a binary mask,
        and returns the mask as a numpy array.
        Args:
            image_path (str): The file path to the PNG image.
        Returns:
            np.ndarray: A binary mask with the same height and width as the input image.
        """
        mask = Image.open(image_path).convert('L')
        return np.array(mask)

    
    def main(self):
        pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./assets', help='path to the video')
    parser.add_argument('--video_name', type=str, default='breakdance', help='path to the video')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--model', type=str, default='spatracker', choices=['spatracker', 'cotracker'], help='model name')
    parser.add_argument('--crop', action='store_true', help='whether to crop the video')
    parser.add_argument('--crop_factor', type=float, default=1, help='whether to crop the video')

    args = parser.parse_args()

    st = SpaTrackerWrapper(root_dir=args.root,
                           video_name=args.video_name,
                           gpu=args.gpu,
                           model=args.model,
                           debug=True)
    st.mkv_to_depth_frames(os.path.join(args.root, args.video_name+'.mkv'))