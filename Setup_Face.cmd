@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if not exist "runtime\python.exe" (
  echo Run install.cmd first.
  pause
  exit /b 1
)
echo Face capture optional setup: MediaPipe Face Landmarker in its own runtime folder.
echo Review the MediaPipe terms: https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE
choice /c YN /n /m "I reviewed the terms. Continue? [Y/N] "
if not "%errorlevel%"=="1" exit /b 1
"runtime\python.exe" -I -X utf8 -B tools\setup_face.py
set "face_result=%errorlevel%"
pause
exit /b %face_result%
