#!/bin/bash

# 初始化仓库
if [ ! -d .git ]; then
  git init
fi

# 生成.gitignore（不忽略自身）
find . -type f -size +50M | sed 's|^\./||' > .gitignore

# 正确添加文件（分步操作）
git rm -r --cached . >/dev/null 2>&1
git add .gitignore            # 先确保添加.gitignore
git add .                     # 再添加其他文件（此时规则已生效）

# 提交
git commit -m "upload folder, skip files over 50MB"

# 添加远程仓库
git remote add origin git@github.com:LiangThree/Prefix.git || true

# 尝试推送
if ! git push -u origin master; then
  echo "检测到历史中存在大文件，开始清理..."
  
  # 备份.gitignore
  cp .gitignore /tmp/safe_gitignore
  
  # 使用正确路径过滤大文件
  git filter-repo \
    --path .gitignore \               # 保留.gitignore
    --path-from-file /tmp/safe_gitignore \  # 按列表删除大文件
    --invert-paths \
    --force
  
  # 恢复.gitignore
  cp /tmp/safe_gitignore .gitignore
  
  # 重新添加文件
  git add .gitignore
  git add .
  git commit -m "清除历史大文件后重新提交"
  
  # 强制推送
  git push -u origin master --force
fi