#!/bin/bash

# Read the list of GitHub links from repos.txt file
while read -r link; do
  echo 'Cloning $link'
  # Extract the repository name from the link
  repo_name=$(basename "$link" .git)
  folder_name="repos/$repo_name"
  # Clone the repository into the repos folder
  git -v clone "$link" "$folder_name"
done < repo_list.txt