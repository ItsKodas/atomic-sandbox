@echo off
rem Atomic Sandbox — build the release app (incremental) and package it in dist\ without launching
rem it. Use start.bat to build and launch from target\. The packaging is scripts\package.ps1 (the
rem release workflow runs the same script), from committed files only: untracked models, personal
rem profiles and models\collections\ never ship.
rem
rem   dist\AtomicSandbox-<version>-windows-x64\        the portable folder to run or hand out
rem       AtomicSandbox.exe, portable.txt              everything the app needs is inside it; its data stays beside the exe
rem       README.md, docs\images\                      the readme and its pictures
rem       scenes\*.scene                               the presets as files (--scene scenes\h2o.scene) and the repo's scenes\ library
rem       models\, profiles\                           the repo's committed models\ (not models\collections\) and profiles\
rem   dist\AtomicSandbox-<version>-windows-x64.zip     the same folder zipped, for a GitHub release
rem   dist\installer-stage\                            what installer\AtomicSandbox.iss packs
rem   dist\AtomicSandbox-Setup-<version>.exe           the installer, when Inno Setup 6 (ISCC.exe) is installed
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

rem the portable folder and zip, the installer's staging folder and (with Inno Setup) the installer;
rem it says what went wrong itself
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\package.ps1" -Exe "target\release\app.exe"
if errorlevel 1 (
    echo.
    echo Packaging failed ^(the reason is above^).
    pause
    exit /b 1
)
endlocal
exit /b 0
