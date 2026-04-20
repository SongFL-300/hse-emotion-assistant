@echo off
setlocal
cd /d %~dp0\..

echo ==========================================
echo   HSE Emotion Chat 环境检测与安装 UI
echo ==========================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] 未找到 python。请先安装 Python 3.10+ 并确保已加入 PATH。
    pause
    exit /b 1
)

python tools\env_installer_ui_v2.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 启动失败，请检查 Python 环境或依赖。
    pause
    exit /b 1
)
endlocal
