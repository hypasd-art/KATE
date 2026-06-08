#!/bin/bash
# Upload KATE project to GitHub repository
# Repository: https://github.com/hypasd-art/KATE

# Navigate to project directory
cd /home/yphao/Experience_Tool/KATE

# Initialize git repository (if not already initialized)
if [ ! -d ".git" ]; then
    git init
    echo "Git repository initialized"
fi

# Add remote repository (if not already added)
if ! git remote | grep -q "origin"; then
    git remote add origin https://github.com/hypasd-art/KATE.git
    echo "Remote origin added"
else
    echo "Remote origin already exists"
fi

# Create .gitignore file to exclude sensitive and unnecessary files
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# Results and Outputs
result/
results/
output/
outputs/
logs/
*.log

# Models (large files)
models/
*.pth
*.bin
*.safetensors
*.ckpt

# Data (optional - uncomment if needed)
# data/
# *.jsonl
# *.csv

# Experience files with embeddings (large files)
*_with_embedding.json

# Temporary files
*.tmp
*.bak
*.swp

# OS
.DS_Store
Thumbs.db
EOF

echo ".gitignore created"

# Stage all files
git add -A

# Show status
git status

# Commit with message
git commit -m "Initial commit: KATE - Knowledge-Augmented Tool-use Enhancement framework

- Add BFCL evaluation framework
- Add AppWorld integration
- Add experience management system
- Add error analysis tools
- Add model handlers for Qwen and other models
- Add documentation (README.md, handler docs)
"

# Push to GitHub
echo ""
echo "Ready to push to GitHub. Choose your authentication method:"
echo "1. HTTPS (will prompt for username/password or token)"
echo "2. SSH (requires SSH key setup)"
echo ""
read -p "Enter choice (1 or 2): " auth_choice

if [ "$auth_choice" = "1" ]; then
    echo "Pushing via HTTPS..."
    git branch -M main
    git push -u origin main --force
elif [ "$auth_choice" = "2" ]; then
    echo "Pushing via SSH..."
    git remote set-url origin git@github.com:hypasd-art/KATE.git
    git branch -M main
    git push -u origin main
else
    echo "Invalid choice. Please run: git push -u origin main"
fi

echo ""
echo "Done! Your repository should be available at:"
echo "https://github.com/hypasd-art/KATE"
