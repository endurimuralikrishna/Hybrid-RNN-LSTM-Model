"""Download locally only after checking source permissions and configuring Kaggle credentials."""
import argparse, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--confirm-permission',action='store_true'); a=p.parse_args()
if not a.confirm_permission: p.error('Verify dataset license, platform terms and ethics requirements first.')
Path('data/raw').mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,'-m','kaggle','datasets','download','-d','infamouscoder/depression-reddit-cleaned','-p','data/raw','--unzip'],check=True)
