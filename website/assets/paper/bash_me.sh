#!/bin/bash
FILES="$@"
for i in $FILES
do
echo "Prcoessing image $i ..."
/usr/bin/convert -thumbnail 200 $i thumb.$i
done
