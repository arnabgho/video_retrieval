import torch
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as T
import torch.nn as nn
import os
import glob
from PIL import Image
import h5py
import deepdish as dd
import os
import cv2
import checksumdir
import json
from annoy import AnnoyIndex
import math

def video_to_frames(video_filename):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_filename)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_rate = int(cap.get(5))
    frames = []
    if cap.isOpened() and video_length > 0:
        frame_ids = frame_rate * list(range(int(video_length / frame_rate)))
        count = 0
        success, image = cap.read()
        while success:
            if count in frame_ids:
                frames.append(image)
            success, image = cap.read()
            count += 1
    return frames


class VideoRepresentation:
    def __init__(self, video_folder):
        self.video_folder = video_folder
        self.video_files = glob.glob(video_folder + "/*.avi") + glob.glob(
            video_folder + "/*.mp4"
        )
        self.video_files = sorted(self.video_files)
        self.total_video_files = len(self.video_files)
        self.annoy_num_trees = 1 + int(math.log(self.total_video_files))
        self.model_conv = torchvision.models.resnet101(pretrained=True)
        if torch.cuda.is_available():
            self.model_conv.cuda()
        self.model_conv.eval()
        self.preprocess_img = self.get_preprocess_img()
        self.temp_folder = "./tmp"
        self.hash_filename = "dir_hash.txt"
        self.annoy_save_name = "index.ann"
        self.map_id_video_filename = "video_index_map.json"

    def get_preprocess_img(self):
        """Preprocessing pipeline for the image"""
        preprocess_img = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return preprocess_img

    def preprocess_image(self, filename):
        """Preprocess the image and prepare it for the network"""
        image_PIL = Image.open(filename)
        image_preprocessed = self.preprocess_img(image_PIL)
        image_preprocessed = torch.unsqueeze(image_preprocessed, 0)
        return image_preprocessed

    def create_input_tensor_from_images(self, video_filename):
        """
        Create the temp directory to store the frames of the video
        Use opencv to extract the frames and write in the tmp folder
        Get the input tensor
        Guaranteed to be less than 30 frames
        Since only 1 frame extracted per second
        """
        self.remove_temporary_folder()
        if not os.path.exists(self.temp_folder):
            os.mkdir(self.temp_folder)
        frames = video_to_frames(video_filename)
        n_frames = len(frames)

        input_tensor = torch.zeros((n_frames, 3, 224, 224))
        for frame_id, frame in enumerate(frames):
            filename = os.path.join(self.temp_folder, "%02d.jpg" % frame_id)
            cv2.imwrite(filename, frame)

            image_preprocessed = self.preprocess_image(filename)

            input_tensor[frame_id] = image_preprocessed

        self.remove_temporary_folder()

        return input_tensor

    def remove_temporary_folder(self):
        """
        Remove files and temporary folder so that
        doesn't corrupt the frames from the next
        video
        """
        if not os.path.exists(self.temp_folder):
            os.mkdir(self.temp_folder)

        files = glob.glob(self.temp_folder + "/*.jpg")

        for f in files:
            os.remove(f)

        os.rmdir(self.temp_folder)

    def write_hash(self):
        """Write hash of directory for further use"""
        dir_hash = checksumdir.dirhash(self.video_folder)
        with open(self.hash_filename, "w") as f:
            f.write(dir_hash)

    def check_recompute_representations(self):
        """
        Check whether the video folder has changed
        If there's some change recompute representations
        """
        if not os.path.exists(self.hash_filename):
            return True
        with open(self.hash_filename, "r") as f:
            loaded_hash = f.read()
        dir_hash = checksumdir.dirhash(self.video_folder)
        return dir_hash != loaded_hash

    def compute_representations(self, save_filename="representations.h5"):
        """
        Compute representations of each of the videos 1 frame per second
        h5py write after each video to not cause memory overflow errors
        Create Annoy index using mean representation of video frames
        Store mapping of annoy ids to video names
        Store representations of each frame of the video indexed by video name
        """
        if self.check_recompute_representations():
            """
            Annoy computation
            """
            embedding_size = 1000
            annoy_index = AnnoyIndex(embedding_size, "angular")

            map_id_video_file = {}

            representation_dictionary = {}
            hf = h5py.File(save_filename, "w")
            for video_file_id, video_file in enumerate(self.video_files):
                video_tensor_preprocessed = self.create_input_tensor_from_images(
                    video_file
                )
                if torch.cuda.is_available():
                    video_tensor_preprocessed = video_tensor_preprocessed.cuda()
                video_features = self.model_conv(video_tensor_preprocessed)
                video_file_key = video_file.split("/")[-1]
                map_id_video_file[video_file_id] = video_file_key
                hf.create_dataset(
                    video_file_key, data=video_features.cpu().detach().numpy()
                )
                """
                Annoy computation
                """
                video_features_mean = video_features.mean(axis=0)
                annoy_index.add_item(
                    video_file_id, video_features_mean.cpu().detach().numpy()
                )
            hf.close()
            self.write_hash()
            annoy_index.build(self.annoy_num_trees)
            annoy_index.save(self.annoy_save_name)
            json.dump(map_id_video_file, open(self.map_id_video_filename, "w"))
        else:
            print("representations don't need recomputation")
