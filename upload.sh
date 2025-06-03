#!/bin/bash

condition=$1

if [ "$condition" == "upload" ]; then
  if [ ! -d .git ]; then
    git init
  fi

  # 创建.gitignore并添加大文件
  find . -type f -size +50M | sed 's|^\./||' > .gitignore
  echo ".gitignore" >> .gitignore  

  # 确保所有被忽略的文件不再被跟踪
  git rm -r --cached . >/dev/null 2>&1

  git add .
  git commit -m "upload folder, skip files over 50MB"
  git remote add origin git@github.com:LiangThree/Prefix.git
  git push -u origin master

elif [ "$condition" == "update" ]; then

  # 初始化仓库（仅在未初始化时执行）
  if [ ! -d .git ]; then
    git init
    git remote add origin git@github.com:LiangThree/Prefix.git
  fi

  # 如果.gitignore不存在，创建一个
  if [ ! -f .gitignore ]; then
    touch .gitignore
  fi

  # 查找并智能添加新的大文件到.gitignore
  find . -type f -size +50M | sed 's|^\./||' | while read -r file; do
    if ! grep -qxF "$file" .gitignore 2>/dev/null; then
      echo "$file" >> .gitignore
      echo "Added to .gitignore: $file"
    fi
  done

  # 添加.gitignore到跟踪文件
  if ! grep -qxF ".gitignore" .gitignore 2>/dev/null; then
    echo ".gitignore" >> .gitignore
  fi

  # 确保所有被忽略的文件不再被跟踪
  git rm -r --cached . >/dev/null 2>&1
  git add .
  
  # 检查是否有需要提交的内容
  if [ -n "$(git status --porcelain)" ]; then
    git commit -m "Update repository [$(date +%Y-%m-%d)]"
    git push -u origin master
  else
    echo "No changes to commit."
  fi

fi