#!/bin/bash

# 确保git filter-repo已安装
if ! command -v git-filter-repo &> /dev/null; then
    echo "安装git-filter-repo..."
    pip install git-filter-repo || {
        echo "安装失败，请手动安装: pip install git-filter-repo"
        exit 1
    }
fi

# 初始化仓库
if [ ! -d .git ]; then
  git init
fi

# 生成.gitignore（忽略大于50MB的文件）
find . -type f -size +50M | sed 's|^\./||' > .gitignore
echo ".gitignore" >> .gitignore  # 防止.gitignore被忽略

# 清除缓存并重新添加文件
git rm -r --cached . >/dev/null 2>&1
git add .gitignore
git add .

# 提交
git commit -m "upload folder, skip files over 50MB" || {
    echo "提交失败，可能没有文件变更"
    exit 1
}

# 添加远程仓库
git remote add origin git@github.com:LiangThree/Prefix.git 2>/dev/null

# 尝试推送
if git push -u origin master; then
    echo "推送成功"
    exit 0
else
    echo "检测到历史中存在大文件，开始清理..."
    
    # 获取所有大文件列表
    find . -type f -size +50M | sed 's|^\./||' > /tmp/big_files.txt
    
    # 使用filter-repo清理历史
    git filter-repo \
        --paths-from-file /tmp/big_files.txt \
        --invert-paths \
        --force
    
    # 重新添加.gitignore
    git add .gitignore
    git add .
    git commit -m "清除历史大文件后重新提交"
    
    # 强制推送
    git push -u origin master --force
fi