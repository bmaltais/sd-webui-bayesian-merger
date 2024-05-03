@echo off

IF NOT EXIST venv (
    echo Creating venv...
    python -m venv venv
)

:: Deactivate the virtual environment to prevent error
call .\venv\Scripts\deactivate.bat

call .\venv\Scripts\activate.bat

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --use-pep517
pip install git+https://github.com/s1dlx/meh.git@sdxl#egg=sd_meh