#!/bin/bash

condition=$1

if [ "$condition" == "upload" ]; then
  if [ ! -d .git ]; then
    git init
  fi

  find . -type f -size +50M | sed 's|^\./||' > .gitignore
  echo ".gitignore" >> .gitignore  

  git add .
  git commit -m "upload folder, skip files over 50MB"
  git remote add origin https://github.com/LiangThree/Prefix.git
  git push -u origin master

elif [ "$condition" == "update" ]; then

  # 初始化仓库（仅在未初始化时执行）
  if [ ! -d .git ]; then
    git init
  fi

  # 智能更新.gitignore（只追加新发现的大文件）
  find . -type f -size +50M | sed 's|^\./||' | while read -r file; do
    if ! grep -qxF "$file" .gitignore 2>/dev/null; then
      echo "$file" >> .gitignore
      echo "Added to .gitignore: $file"
    fi
  done

  # 添加所有未忽略文件（包括.gitignore本身）
  git add .

  # 提交更新
  git commit -m "Update repository [$(date +%Y-%m-%d)]"

  # 确保远程仓库配置正确
  git remote rm origin 
  git remote add origin git@github.com:LiangThree/Prefix.git

  # 推送更新到主分支
  git push -u origin master

fi