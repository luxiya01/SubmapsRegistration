#!/bin/bash

cd ../bin
folder="mission11-subset-test2-more-overlap"
input="$folder/loop_closure_pairs.txt"
while IFS=" " read -r col1 col2
do
    ./gicp_registration --submap1 "$folder/submap_$col1.pcd"\
                        --submap2 "$folder/submap_$col2.pcd"\
                        --config ../config.yaml
done < $input
