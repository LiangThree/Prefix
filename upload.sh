#!/bin/bash

# 定义要删除的大文件路径模式（匹配.gitignore中的大文件）
# 注意：我们将使用.gitignore中的大文件路径来删除历史记录
LARGE_FILES="large_files_list.txt"

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