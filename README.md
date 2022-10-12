# A non-native English corpus for phrasing-break-assessment
## Introduction
This corpus aims to provide a free public dataset for the phrasing break assessment task. we collected 748 audio samples from Chinese ESL learners. Then three linguists are invited to assess them with overall performance on phrase break and each individual phrasing breaks ranging from 1 (Poor), 2 (Fair), and 3 (Great). 

## The scoring metric
The linguists score at two levels: overall-sentence-level and single-break-level.

### overall-sentence-level
Assess the overall performance on phrase breaks in the sentence.

Score range: 1-3

- 3: all phrase breaks in the sentence are correct
- 2: most of phrase breaks in the sentence are correct but some need to be improved
- 1: most of phrase breaks in the sentence are incorrect
### single-break-level
Assess each individual phrasing breaks in the sentence.

Score range: 1-3

- 3: the phrase break is perfect
- 2: the phrase break is fair and can to be improved
- 1: the phrase break is wrong

## Sentence distribution
We collected 800 utterances of 18 different sentences of various lengths, and the detail distribution is shown in the following table. 
| No.   | Sentence                                                                                                                    | Num |
| ----- | --------------------------------------------------------------------------------------------------------------------------- | --- |
| 1     | Their poor parents were just starving as usual.                                                                             | 33  |
| 2     | One afternoon, he thought up a good plan to have fun.                                                                       | 42  |
| 3     | Cinderella danced so joyfully that she forgot the time.                                                                     | 32  |
| 4     | They made a choice and went out in search of food.                                                                          | 43  |
| 5     | Plenty of complicated thoughts about the rainbow have been formed.                                                          | 36  |
| 6     | The following week, the same policeman sees the same man with the tiger again.                                              | 54  |
| 7     | His mother noticed his left shoe was on his right foot.                                                                     | 28  |
| 8     | One spring day some naughty boys were playing near a pond.                                                                  | 41  |
| 9     | Our class is going to put on a show on may seventh.                                                                         | 37  |
| 10    | Now, can you give me a few words to describe the body shape of this girl?                                                   | 48  |
| 11    | The man replied, "i did we had such a good time we are going to the beach this weekend!"                                    | 62  |
| 12    | His wife says, "no problem but i have to find all the money you have hidden from me.                                        | 50  |
| 13    | There was once a poor boy, and he watched his sheep in the fields next to a dark forest.                                    | 42  |
| 14    | The frogs in the pond were very afraid of the children because the stones killed some of them.                              | 39  |
| 15    | One day a crow stood on a branch near his nest and felt very happy with the meat in his mouth.                              | 44  |
| 16    | But the mouse finally got out of her mouth and ran away since she could not bite it.                                        | 49  |
| 17    | Then the old woman became very angry at the cat and began to hit her for she had not killed the mouse.                      | 29  |
| 18    | Until the fox thought highly of the crow's beautiful voice, the crow felt quite proud and finally opened his mouth to sing. | 39  |
| total | 748|


## Data structure

The following tree shows the file structure of this corpus:
```
├───cross-validation-testset
│   ├───coarse_grain_res.csv
│   ├───fine_grain_res.csv
│   ├───coarse_grain_set
│   │       coarse_grain_testset_0.csv
│   │       coarse_grain_testset_1.csv
│   │       coarse_grain_testset_2.csv
│   │       coarse_grain_testset_3.csv
│   │       coarse_grain_testset_4.csv
│   │       coarse_grain_trainset_0.csv
│   │       coarse_grain_trainset_1.csv
│   │       coarse_grain_trainset_2.csv
│   │       coarse_grain_trainset_3.csv
│   │       coarse_grain_trainset_4.csv
│   │
│   └───fine_grain_set
│           fine_grain_testset_0.csv
│           fine_grain_testset_1.csv
│           fine_grain_testset_2.csv
│           fine_grain_testset_3.csv
│           fine_grain_testset_4.csv
│           fine_grain_trainset_0.csv
│           fine_grain_trainset_1.csv
│           fine_grain_trainset_2.csv
│           fine_grain_trainset_3.csv
│           fine_grain_trainset_4.csv
│
├───resource
│   ├───A-res_form
│   ├───B-res_form
│   └───C-res_form
├───wavs
└───metadata.csv
```
The `wavs` folder contains our all wavs resource and match the `metadata.csv`.

The `resource` folder contains the scores from three linguists label A, B and C according to three subfolders.

The `cross-validation-testset` folder contains trainset and testset in fine-tune step.

## Citation
Please cite our paper if you find this work useful:

```
...
```

