@echo off
rem Atomic Sandbox — build the release app without launching it (incremental).
rem Use start.bat to build and launch.
setlocal
cd /d "%~dp0"

set "CARGO=cargo"
where cargo >nul 2>nul || set "CARGO=%USERPROFILE%\.cargo\bin\cargo.exe"
if not exist "%CARGO%" if not "%CARGO%"=="cargo" (
    echo Could not find cargo. Install Rust from https://rustup.rs and try again.
    pause
    exit /b 1
)

"%CARGO%" build -p app --release
if errorlevel 1 (
    echo.
    echo The build failed. If an Atomic Sandbox window is already open, close it and run this again
    echo ^(a running app locks its own executable^). Otherwise the error is printed above.
    pause
    exit /b 1
)
echo Built: target\release\app.exe
endlocal
