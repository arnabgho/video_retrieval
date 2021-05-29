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
from annoy import AnnoyIndex
import json

"""
LSH:
Build representation of the video by averaging out the representations of
the frames of the video

Build an index of these averaged out representations serving as a representation of the video

Query the representation of the image using this index for fast retrieval
Compare the representation of the frames of the image to obtain best timestamp match
"""

class LSHComparison:
    def __init__(
        self,
        representation_file="representations.h5",
        index_file="index.ann",
        map_id_video_filename="video_index_map.json",
    ):
        self.representation_file = representation_file
        self.index_file = index_file
        self.map_id_video_filename = map_id_video_filename
        self.model_conv = torchvision.models.resnet101(pretrained=True)
        if torch.cuda.is_available():
            self.model_conv.cuda()
        self.model_conv.eval()
        self.preprocess_img = self.get_preprocess_img()
        self.index = AnnoyIndex(1000, "angular")
        self.index.load(self.index_file)
        self.map_id_video_file = json.load(open(self.map_id_video_filename))

    def get_preprocess_img(self):
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
        image_PIL = Image.open(filename)
        image_preprocessed = self.preprocess_img(image_PIL)
        image_preprocessed = torch.unsqueeze(image_preprocessed, 0)
        return image_preprocessed

    def query_image(self, query_image_filename, topk):
        # get the query representation
        image_preprocessed = self.preprocess_image(query_image_filename)
        if torch.cuda.is_available():
            image_preprocessed = image_preprocessed.cuda()

        image_features = self.model_conv(image_preprocessed)
        image_features_numpy = image_features.cpu().detach().numpy()
        image_features_annoy_search = image_features_numpy.squeeze(axis=0)
        (
            nearest_neighbor_indices,
            nearest_neighbor_scores,
        ) = self.index.get_nns_by_vector(
            image_features_annoy_search, topk, search_k=-1, include_distances=True
        )
        similarity_scores = {}
        similarity_max_times = {}
        with h5py.File(self.representation_file, "r") as f:
            for nearest_neighbor_index in nearest_neighbor_indices:
                video_id = self.map_id_video_file[str(nearest_neighbor_index)]
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
        descending_similarity, similarity_max_times = self.query_image(
            query_image_filename, topk
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
