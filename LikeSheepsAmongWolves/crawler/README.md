# Direct Unbiased Random Walk Crawler

- This folder contains a random-walk based crawler on Twitter implemented using Neo4j graph database. 
We use Neo4j v. 3.2.6 community edition.

- The sampling algorithm we implement comes from the paper 
[Sampling directed graphs with random walks](1) 
from Ribeiro et al.

- Notice that the we assume some .json files with authentication for twitter and neo4j, 
those are placed in the `../secrets/` folder.

[1]:http://ieeexplore.ieee.org/abstract/document/6195540/