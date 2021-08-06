# Twitter-Algorithmic-Bias-Challenge-C4T
Repository for Code4Thought's participation in Twitter's Algorithmic Bias challenge


# Steps to reproduce experiments
### Dataset
* Download [FairFace dataset](https://github.com/joojs/fairface) with Padding=1.25 from [here](https://drive.google.com/file/d/1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL/view)
* Unzip .zip file in `data` folder of this repository

In a terminal:
1. Install poetry with: `pip install poetry`
2. `poetry install`
3. Depending on the experiment you want run: 
    * Male - Female comparison: <br> `poetry run inv single-comparison --trait1 gender --trait2 gender --group1-name Male --group2-name Female --n-samples 10000` 
    * White - Black comparison: <br>` poetry run inv single-comparison --trait1 race --trait2 race --group1-name White --group2-name Black --n-samples 10000`
    * White Males - Black Males comparison: <br>`poetry run inv double-comparison --race-1 White --race-2 Black --gender-1 Male --gender-2 Male --n-samples 10000`

For custom experiments run:
* Comparing two single groups: <br> `poetry run inv single-comparison --trait1 <trait_name1> --trait2 <trait_name2> --group1-name <group_name1> --group2-name <group_name2> --n-samples <n_samples>`
* Comparing two double groups containg a gender and a race:  <br> `poetry run inv double-comparison --race-1 <race_group1> --race-2 <race_group2> --gender-1 <gender_group1> --gender-2 <gender_group2> --n-samples <n_samples>`
