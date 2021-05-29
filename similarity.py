import sys

sys.path.append(".")
import argparse
from create_video_representations_from_images import VideoRepresentation
from brute_force_comparison import BruteForce
from lsh_comparison import LSHComparison

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--topk", type=int, help="number of top k integers to be displayed")
parser.add_argument("--image", type=str, help="the image being searched for")
parser.add_argument(
    "--type",
    type=str,
    default="lsh",
    help="the type of nearest neighbor search to perform, either lsh or brute_force",
)
parser.add_argument(
    "--video_folder", type=str, help="the video directory containing all the files"
)
args = parser.parse_args()

"""
First compute the representations of each frame in the videos in the video_folder
Only compute representations if the hash of the folder doesn't match
Save representations as .h5 file for further processing

Build indices using annoy for fast nearest neighbor lookup
Might need to tweak annoy hyperparameters for large scale data

Brute Force:
Compare the representation of the query image to all the frame representations
return the video containing the frame with the highest similarity

LSH:
Build representation of the video by averaging out the representations of
the frames of the video

Build an index of these averaged out representations serving as a representation of the video

Query the representation of the image using this index for fast retrieval
Compare the representation of the frames of the image to obtain best timestamp match
"""

video_representation = VideoRepresentation(args.video_folder)
video_representation.compute_representations()

if args.type == "brute_force":
    brute_force = BruteForce()
    result = brute_force.query_top_k(args.image, args.topk)
elif args.type == "lsh":
    lsh = LSHComparison()
    result = lsh.query_top_k(args.image, args.topk)

print(result)
