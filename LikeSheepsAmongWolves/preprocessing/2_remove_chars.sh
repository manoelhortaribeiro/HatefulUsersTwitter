#!/usr/bin/env bash

perl -CSDA -pe 's/[^\x9\xA\xD\x20-\x{D7FF}\x{E000}-\x{FFFD}\x{10000}-\x{10FFFF}]+//g;' ../data/preprocessing/users.graphml > ../data/preprocessing/users2.graphml
rm ../data/preprocessing/users.graphml
mv ../data/preprocessing/users2.graphml ../data/preprocessing/users.graphml