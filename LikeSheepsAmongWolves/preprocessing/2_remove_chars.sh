#!/usr/bin/env bash

perl -CSDA -pe 's/[^\x9\xA\xD\x20-\x{D7FF}\x{E000}-\x{FFFD}\x{10000}-\x{10FFFF}]+//g;' ../data/users.graphml > ../data/users2.graphml
rm ../data/users.graphml
mv ../data/users2.graphml ../data/users.graphml