@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
echo InstantHMR live estimator setup. Close RGB MoCap before continuing.
echo This adds ONNX Runtime GPU (about 0.4 GiB) in its own folder; no runtime is changed.
echo The model files are downloaded separately in Model Manager.
echo InstantHMR code is Apache-2.0. The medium, small, tiny and nano weights are CC BY-NC 4.0
echo (non-commercial); the medium-commercial weights are Apache-2.0.
echo See https://github.com/Naturalpad-studio/InstantHMR before relying on any weight.
if not exist "runtime\python.exe" (
  echo Run install.cmd first.
  pause
  exit /b 1
)
"runtime\python.exe" -I -X utf8 -B tools\setup_instanthmr.py
set "instant_result=%errorlevel%"
pause
exit /b %instant_result%
