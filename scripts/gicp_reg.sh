#!/bin/bash

cd ../bin

#folder="230125-mission11-navlab-smooth"
#folder="230210-mission11-navlab-smooth-map"
folder="230417-submaps-with-max-overlap"
input="$folder/loop_closures_pairs.txt"
while IFS=" " read -r col1 col2
do
    ./gicp_registration --submap1 "$folder/submap_$col1.pcd"\
                        --submap2 "$folder/submap_$col2.pcd"\
                        --config ../gicp.yaml
done < $input
