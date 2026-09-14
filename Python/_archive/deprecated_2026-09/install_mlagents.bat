@echo off
echo ML-Agents 환경 설정 시작...

REM Python 가상환경 생성 (선택사항)
REM python -m venv venv
REM call venv\Scripts\activate

REM 기존 ML-Agents 제거
pip uninstall -y mlagents mlagents-envs

REM 호환되는 버전 설치 (Unity release_23과 호환)
pip install --upgrade pip
pip install numpy==1.23.5
pip install mlagents==0.30.0
pip install protobuf==3.20.3
pip install torch torchvision torchaudio

echo.
echo 설치 완료! 설치된 버전:
pip list | findstr mlagents
pip list | findstr numpy
pip list | findstr protobuf
pip list | findstr torch

echo.
echo Unity와 연결하려면 Python 스크립트를 먼저 실행하세요:
echo python main.py

pause