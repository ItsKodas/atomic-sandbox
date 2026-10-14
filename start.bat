@echo off
rem Atomic Sandbox — build (release, incremental) and launch.
rem Double-click to start, or run from a terminal with options, e.g.:
rem   start.bat --preset h2o --quality balanced
rem   start.bat --backend cpu
rem Options are passed straight to the app (see README "Running it").
setlocal
cd /d "%~dp0"

set "CARGO=cargo"
where cargo >nul 2>nul || set "CARGO=%USERPROFILE%\.cargo\bin\cargo.exe"
if not exist "%CARGO%" if not "%CARGO%"=="cargo" (
    echo Could not find cargo. Install Rust from https://rustup.rs and try again.
    pause
    exit /b 1
)

echo Building Atomic Sandbox (release)...
"%CARGO%" build -p app --release
if errorlevel 1 (
    echo.
    echo The build failed. If an Atomic Sandbox window is already open, close it and run this again
    echo ^(a running app locks its own executable^). Otherwise the error is printed above.
    pause
    exit /b 1
)

echo Launching...
"%~dp0target\release\app.exe" %*
if errorlevel 1 pause
endlocal
