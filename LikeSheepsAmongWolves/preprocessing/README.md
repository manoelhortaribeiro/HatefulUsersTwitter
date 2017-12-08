# From Database to Annotation Ready

![](../imgs/preprocessing.png)


- This folder contains the scripts (and a python notebook with the commented scripts) 
that can be used to transform the data from the neo4j graph to easier to use files.

- A brief description of the scripts:

    - `1_get_user_graph.py` simply extracts the graph of users from the neo4j dataset. 
    These are saved in a file `users.graphml` in a networkx-readable format.
    
    - `2_get_user_graph.py` removes some annoying characters present in the tweets 
    that are not allowed in the `.graphml` format.
    
    - `3_get_tweets_table.py` gets a table with all the tweets from the users. 
    The script is a bit overly complex, because the field separator chosen was dumb.
    
    - `4_get_infected_user_graph.py` reads the words in our lexicon in `../data/lexicon.txt`
    and tries to match them in each tweet. We mark users who used the words as "infected".
    
    - `5_get_diffusion_graph.py` implements the a diffusion based on
     DeGroot's model proposed by Golub & Jackson (2010).
    
    - `6_get_users_to_annotate.py` gets users to annotate from all strata.
    