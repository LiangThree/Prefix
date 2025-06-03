#!/bin/bash

condition=$1

# 定义要删除的大文件路径模式（匹配.gitignore中的大文件）
# 注意：我们将使用.gitignore中的大文件路径来删除历史记录
LARGE_FILES="large_files_list.txt"

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

  # 提交
  git commit -m "upload folder, skip files over 50MB"

  # 添加远程仓库
  git remote add origin git@github.com:LiangThree/Prefix.git

  # 尝试推送
  if ! git push -u origin master; then
    echo "Push failed, possibly due to large files in history. Attempting to filter out large files..."
    # 备份.gitignore，因为下面会删除整个仓库的重新初始化
    cp .gitignore /tmp/safe_gitignore
    # 使用git filter-repo删除历史中的大文件
    # 注意：这将重写历史，删除所有匹配大文件
    git filter-repo --path .gitignore --invert-paths --force
    # 恢复.gitignore
    cat /tmp/safe_gitignore > .gitignore
    # 重新添加远程仓库
    git remote add origin git@github.com:LiangThree/Prefix.git
    # 强制推送
    git push -u origin master --force
  fi

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

  # 添加.gitignore自忽略
  if ! grep -qxF ".gitignore" .gitignore 2>/dev/null; then
    echo ".gitignore" >> .gitignore
  fi

  # 清除缓存
  git rm -r --cached . >/dev/null 2>&1
  git add .

  # 提交更新
  if [ -n "$(git status --porcelain)" ]; then
    git commit -m "Update repository [$(date +%Y-%m-%d)]"

    # 尝试推送
    if ! git push -u origin master; then
      echo "Push failed, possibly due to large files in history. Attempting to filter out large files..."
      # 备份.gitignore
      cp .gitignore /tmp/safe_gitignore
      # 提取当前提交的.gitignore中所有大文件路径（作为要删除的历史文件）
      # 注意：这里我们假设.gitignore中除了注释和空行，其他都是大文件路径
      # 但注意.gitignore中可能有通配符，git filter-repo不支持通配符，所以这里只能按行来，每行一个路径
      # 使用git filter-repo删除历史中出现的所有大文件
      # 准备一个文件列表，包含所有在.gitignore中的文件（排除注释和空行）
      grep -v '^#' .gitignore | grep -v '^\s*$' > /tmp/large_files_list.txt
      # 使用文件列表来过滤
      git filter-repo --paths-from-file /tmp/large_files_list.txt --invert-paths --force
      # 恢复.gitignore
      cat /tmp/safe_gitignore > .gitignore
      git add .gitignore
      git commit -m "Remove large files from history"
      git remote add origin git@github.com:LiangThree/Prefix.git
      git push -u origin master --force
    fi
  else
    echo "No changes to commit."
  fi

fi