#### Trying to obtain the right timestamp by using the extracted frame from the video

```
python similarity.py --image './test_images/video_8.jpg' --video_folder './CV_Challenge_data/videos/' --topk 2 --type lsh
```

```
{'results': [{'video': 'video.mp4', 'similarity_score': 1.0000001, 'time_in_seconds': 9}, {'video': 'v_Biking_g02_c03.avi', 'similarity_score': 0.66885936, 'time_in_seconds': 8}]}
```

#### Fencing image from the wild

```
python similarity.py --image './test_images/fencing.jpg' --video_folder './CV_Challenge_data/videos/' --topk 2 --type lsh
```

```
{'results': [{'video': 'v_Fencing_g02_c02.avi', 'similarity_score': 0.6070603, 'time_in_seconds': 3}, {'video': 'v_PlayingGuitar_g03_c07.avi', 'similarity_score': 0.5055058, 'time_in_seconds': 3}]}
```

#### Guitar image from the wild

```
python similarity.py --image './test_images/guitar.jpeg' --video_folder './CV_Challenge_data/videos/' --topk 2 --type lsh
```

```
{'results': [{'video': 'v_PlayingGuitar_g04_c01.avi', 'similarity_score': 0.793892, 'time_in_seconds': 2}, {'video': 'v_PlayingGuitar_g03_c07.avi', 'similarity_score': 0.73543864, 'time_in_seconds': 5}]}
```

#### Failure case: Horse racing image from the wild
```
python similarity.py --image './test_images/horse_races.jpg' --video_folder './CV_Challenge_data/videos/' --topk 2 --type lsh
```
```
{'results': [{'video': 'v_Biking_g01_c01.avi', 'similarity_score': 0.51946926, 'time_in_seconds': 5}, {'video': 'v_Fencing_g02_c02.avi', 'similarity_score': 0.38467622, 'time_in_seconds': 3}]}
```

#### Another Horse racing image from the wild
```
python similarity.py --image './test_images/hurdle-horse-race.jpg' --video_folder './CV_Challenge_data/videos/' --topk 2 --type lsh
```

```
{'results': [{'video': 'v_HorseRace_g05_c01.avi', 'similarity_score': 0.63814604, 'time_in_seconds': 7}, {'video': 'v_Biking_g01_c01.avi', 'similarity_score': 0.6268203, 'time_in_seconds': 5}]}
```

#### Walking dog image from the wild

```
python similarity.py --image './test_images/walking_dog.jpg' --video_folder './CV_Challenge_data/videos/' --topk 2 --type lsh
```
```
{'results': [{'video': 'v_WalkingWithDog_g19_c01.avi', 'similarity_score': 0.49403825, 'time_in_seconds': 6}, {'video': 'v_WalkingWithDog_g17_c03.avi', 'similarity_score': 0.36703485, 'time_in_seconds': 1}]}
```

#### Testing with negative value of topk Walking dog image from the wild

```
python similarity.py --image './test_images/walking_dog.jpg' --video_folder './CV_Challenge_data/videos/' --topk -1 --type lsh
```
```
{'results': []}
```

#### Testing with value of topk = num_videos Walking dog image from the wild

```
python similarity.py --image './test_images/walking_dog.jpg' --video_folder './CV_Challenge_data/videos/' --topk 11 --type lsh
```
```
{'results': [{'video': 'v_WalkingWithDog_g19_c01.avi', 'similarity_score': 0.49403825, 'time_in_seconds': 6}, {'video': 'v_Fencing_g02_c02.avi', 'similarity_score': 0.37865192, 'time_in_seconds': 4}, {'video': 'v_WalkingWithDog_g17_c03.avi', 'similarity_score': 0.36703485, 'time_in_seconds': 1}, {'video': 'v_Fencing_g04_c01.avi', 'similarity_score': 0.36606097, 'time_in_seconds': 3}, {'video': 'v_HorseRace_g05_c01.avi', 'similarity_score': 0.22919872, 'time_in_seconds': 5}, {'video': 'v_Biking_g02_c03.avi', 'similarity_score': 0.2031042, 'time_in_seconds': 8}, {'video': 'v_Biking_g01_c01.avi', 'similarity_score': 0.1753347, 'time_in_seconds': 3}, {'video': 'video.mp4', 'similarity_score': 0.15344115, 'time_in_seconds': 11}, {'video': 'v_PlayingGuitar_g04_c01.avi', 'similarity_score': 0.1502943, 'time_in_seconds': 2}, {'video': 'v_PlayingGuitar_g03_c07.avi', 'similarity_score': 0.14838652, 'time_in_seconds': 9}, {'video': 'v_HorseRace_g04_c04.avi', 'similarity_score': 0.026506688, 'time_in_seconds': 4}]}
```

#### Testing with value of topk >>> num_videos Walking dog image from the wild

```
python similarity.py --image './test_images/walking_dog.jpg' --video_folder './CV_Challenge_data/videos/' --topk 1100 --type lsh
```
```
{'results': [{'video': 'v_WalkingWithDog_g19_c01.avi', 'similarity_score': 0.49403825, 'time_in_seconds': 6}, {'video': 'v_Fencing_g02_c02.avi', 'similarity_score': 0.37865192, 'time_in_seconds': 4}, {'video': 'v_WalkingWithDog_g17_c03.avi', 'similarity_score': 0.36703485, 'time_in_seconds': 1}, {'video': 'v_Fencing_g04_c01.avi', 'similarity_score': 0.36606097, 'time_in_seconds': 3}, {'video': 'v_HorseRace_g05_c01.avi', 'similarity_score': 0.22919872, 'time_in_seconds': 5}, {'video': 'v_Biking_g02_c03.avi', 'similarity_score': 0.2031042, 'time_in_seconds': 8}, {'video': 'v_Biking_g01_c01.avi', 'similarity_score': 0.1753347, 'time_in_seconds': 3}, {'video': 'video.mp4', 'similarity_score': 0.15344115, 'time_in_seconds': 11}, {'video': 'v_PlayingGuitar_g04_c01.avi', 'similarity_score': 0.1502943, 'time_in_seconds': 2}, {'video': 'v_PlayingGuitar_g03_c07.avi', 'similarity_score': 0.14838652, 'time_in_seconds': 9}, {'video': 'v_HorseRace_g04_c04.avi', 'similarity_score': 0.026506688, 'time_in_seconds': 4}]}
```
