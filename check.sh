echo "- OS information: `uname -mrsv`"
python3 << EOF
import sys, espnet, torch
pyversion = sys.version.replace('\n', ' ')
print(f"""- python version: \`{pyversion}\`
- espnet version: \`espnet {espnet.__version__}\`
- pytorch version: \`pytorch {torch.__version__}\`""")
EOF
cat << EOF
- Git hash: \`$(git rev-parse HEAD)\`
  - Commit date: \`$(git log -1 --format='%cd')\`
EOF