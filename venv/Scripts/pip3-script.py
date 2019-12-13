#!F:\Research\captain\Captain小组项目本子\深度学习遥感图像质量检测\深度学习质量自动检测项目中期检查\清晰度评价\venv\Scripts\python.exe -x
# EASY-INSTALL-ENTRY-SCRIPT: 'pip==9.0.1','console_scripts','pip3'
__requires__ = 'pip==9.0.1'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('pip==9.0.1', 'console_scripts', 'pip3')()
    )
