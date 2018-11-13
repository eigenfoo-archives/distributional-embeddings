# Remove HTML doc tags
cat wiki_* | sed '/<doc/d' | sed '/<\/doc>/d' > data.txt

# Strip punctuation and remove single-word lines, which indicate titles
sed -i 's/\./$keep/g; s/[[:punct:]]//g; s/keep/\./g; /^[A-Za-z][A-Za-z]*$/d' data.txt

# Lowercase
cat data.txt | tr '[[:upper:]]' '[[:lower:]]' > foo.txt; rm data.txt; mv foo.txt data.txt
