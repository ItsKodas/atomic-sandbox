@echo off
rem Atomic Sandbox — build the release app (incremental) and lay out a runnable copy in dist\
rem without launching it. Use start.bat to build and launch from target\.
rem
rem   dist\AtomicSandbox-<version>-windows-x64\        the folder to run or hand out
rem       AtomicSandbox.exe                            everything the app needs is inside it
rem       README.md, docs\images\                      the readme and its pictures
rem       scenes\*.scene                               the presets as files (--scene scenes\h2o.scene)
rem       models\                                     the repo's models\ (trained models, pre-downloaded); Save/Load model start there
rem   dist\AtomicSandbox-<version>-windows-x64.zip     the same folder zipped, for a GitHub release
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

rem the workspace version, from the root Cargo.toml's  version = "x.y.z"
set "VERSION="
for /f tokens^=2^ delims^=^" %%v in ('findstr /b /c:"version = " Cargo.toml') do set "VERSION=%%v"
if not defined VERSION set "VERSION=dev"
set "NAME=AtomicSandbox-%VERSION%-windows-x64"
set "OUT=dist\%NAME%"

if exist "%OUT%" rmdir /s /q "%OUT%" 2>nul
if exist "%OUT%\AtomicSandbox.exe" (
    echo.
    echo Could not replace %OUT%: close the Atomic Sandbox started from dist\ and run this again.
    pause
    exit /b 1
)
if exist "dist\%NAME%.zip" del /q "dist\%NAME%.zip"

mkdir "%OUT%\scenes" "%OUT%\models" "%OUT%\docs\images" >nul
xcopy /e /i /q /y "models" "%OUT%\models" >nul || goto :copyfail
copy /y "target\release\app.exe" "%OUT%\AtomicSandbox.exe" >nul || goto :copyfail
copy /y "README.md" "%OUT%\" >nul || goto :copyfail
if exist "LICENSE" copy /y "LICENSE" "%OUT%\" >nul
copy /y "crates\app\presets\*.scene" "%OUT%\scenes\" >nul || goto :copyfail
xcopy /e /i /q /y "docs\images" "%OUT%\docs\images" >nul || goto :copyfail

rem a few tries: a virus scan of the just-copied exe can hold it for a moment
powershell -NoProfile -Command "for ($i = 0; $i -lt 5; $i++) { try { Compress-Archive -Path '%OUT%' -DestinationPath 'dist\%NAME%.zip' -Force -ErrorAction Stop; exit 0 } catch { Start-Sleep -Seconds 2 } }; exit 1"
if errorlevel 1 (
    echo The folder is ready but zipping it failed: %OUT%
    pause
    exit /b 1
)
echo Dist:   %OUT%\AtomicSandbox.exe
echo Zip:    dist\%NAME%.zip
endlocal
exit /b 0

:copyfail
echo.
echo Copying into %OUT% failed ^(the error is above^).
pause
exit /b 1
