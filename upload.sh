#!/bin/bash

if [ ! -d .git ]; then
  git init
fi

find . -type f -size +50M | sed 's|^\./||' > .gitignore
echo ".gitignore" >> .gitignore  

git add .
git commit -m "upload folder, skip files over 50MB"
git remote add origin https://github.com/LiangThree/MyProject.git
git push -u origin master