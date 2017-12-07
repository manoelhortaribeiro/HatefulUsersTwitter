# Like Sheep Among Wolves

## Folder Structure

There is a loose sense of "order" in the folders, although some are only auxiliary:

- `./crawler/` contains the code used to extract the dataset. You need to set neo4j to run it.

- `./prepreprocessing/` contains scripts to select the users to be annotated, and extract their tweets.

- `./features/` contains scripts to get the features to be analyzed and that will be fed into the classifier.

- `./analysis/`

- `./classification/`

- Auxiliary folders:

    - `./data/`
    
    - `./secrets/`
    
    - `./tmp/`

    - `./img/`
    
# Reproducibility

Some of the files used are not shared due to impracticality or because sharing them violates Twitter's guidelines. 
However, we do share many of our files:

-   `bad_words.txt`
-   `lexicon.txt`
