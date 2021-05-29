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
from sklearn.metrics.pairwise import cosine_similarity
import os
import cv2
import numpy as np

"""
Brute Force:
Compare the representation of the query image to all the frame representations
return the video containing the frame with the highest similarity
"""

class BruteForce:
    def __init__(self, representation_file="representations.h5"):
        self.representation_file = representation_file
        self.model_conv = torchvision.models.resnet101(pretrained=True)
        if torch.cuda.is_available():
            self.model_conv.cuda()
        self.model_conv.eval()
        self.preprocess_img = self.get_preprocess_img()

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

    def query_image(self, query_image_filename):
        """
        Get the representation of the query image
        Compare the representations to all the frames in the h5 file
        Obtain the maximum frame match in each video
        Also indicates which second leads to the maximum match
        Return sorted list, sorted in descending order of similarity
        """
        image_preprocessed = self.preprocess_image(query_image_filename)
        if torch.cuda.is_available():
            image_preprocessed = image_preprocessed.cuda()
        image_features = self.model_conv(image_preprocessed)
        image_features_numpy = image_features.cpu().detach().numpy()
        similarity_scores = {}
        similarity_max_times = {}
        with h5py.File(self.representation_file, "r") as f:
            for video_id in f.keys():
                video_representation = f[video_id]
                cosine_value = cosine_similarity(
                    video_representation, image_features_numpy
                )
                best_match_time_in_seconds = np.argmax(cosine_value)
                similarity_scores[video_id] = cosine_value[best_match_time_in_seconds]
                similarity_max_times[video_id] = best_match_time_in_seconds + 1
        descending_similarity = sorted(
            similarity_scores.items(), key=lambda x: x[1], reverse=True
        )
        return descending_similarity, similarity_max_times

    def query_top_k(self, query_image_filename, topk):
        """
        Use the sorted list provided by query_image to pick the topk
        Prepare the final dictionary object to print the exact time
        alongside the similarity score
        """
        descending_similarity, similarity_max_times = self.query_image(
            query_image_filename
        )
        if topk > len(descending_similarity):
            topk = len(descending_similarity)
        final_top_k_results = {}
        descending_list = []
        for k in range(topk):
            detail_dict = {}
            detail_dict["video"] = descending_similarity[k][0]
            detail_dict["similarity_score"] = descending_similarity[k][1][0]
            detail_dict["time_in_seconds"] = similarity_max_times[detail_dict["video"]]
            descending_list.append(detail_dict)
        final_top_k_results["results"] = descending_list
        return final_top_k_results
