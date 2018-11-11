# Counts the number of unique words in data.txt
# From https://stackoverflow.com/a/34377954
tr ':;,?!\"' ' ' < data.txt | tr -s ' ' '\n' | awk '!a[$0]++{c++} END{print c}'
