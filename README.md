## Diagrammatic overview of the system architecture
<img src='Architecture.jpg' width=700>


## Prerequisites
- Linux or macOS
- Python 3
- OpenCV

## Getting Started
- Clone this repo:
```
git clone https://github.com/arnabgho/video_retrieval
cd video_retrieval
```
- Install PyTorch 1.0+ and dependencies from http://pytorch.org
- Install Torchvision
- Install all requirements
```
conda create -n 'video_retrieval' python=3.6
conda activate video_retrieval
pip install -r requirements.txt
```

### Run the similarity command

```
python similarity.py --image './CV_Challenge_data/images/biking.jpg' --video_folder './CV_Challenge_data/videos/' --topk 2
```
### Go to interesting_tests.md for more example test cases and edge cases

### Design comments:

In terms of extensibility, the solution should scale to 10^4 videos as long as the disk space doesn't overflow
If the number of videos goes beyond 10^4 to store the representations a distributed key value store such as Cassandra might be applicable
This solution relies on building of the annoy index on the mean representations of all the frames of the video. The tradeoff being the index being not quite flexible. Thus, this index has to be computed as a batch job if the contents of the video folder changes quite frequently. Can perhaps be hourly or daily depending on the flux of the change of the video folder. 
