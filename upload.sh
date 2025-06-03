#!/bin/bash

# GitHub上传助手
# 功能：跳过超过50MB的文件，防止上传失败
# 使用：在项目目录运行，首次使用需配置GitHub仓库

# 检查是否在Git仓库中
git init

# 查找超过50MB的文件（包含未跟踪文件）
large_files=$(find . -type f -size +50M -not -path "./.git/*")

if [ -n "$large_files" ]; then
    echo "发现超过50MB的文件:"
    echo "$large_files"
    
    # 添加规则到.gitignore
    for file in $large_files; do
        # 确保路径从项目根开始
        rel_path=${file#./}
        if ! grep -q "$rel_path" .gitignore 2>/dev/null; then
            echo "添加规则: $rel_path"
            echo "/$rel_path" >> .gitignore
        fi
    done
fi

# 添加所有跟踪文件
git add .

# 配置提交信息
if [ -z "$(git status --porcelain)" ]; then
    echo "没有需要提交的内容"
    exit 0
fi

git commit -m "update"

# 配置远程仓库
if [ -z "$(git remote)" ]; then
    git remote add origin "https://github.com/LiangThree/Prefix.git"
fi

# 推送代码（强制设置上游分支）
current_branch=$(git branch --show-current)
git push -u origin "$current_branch"
