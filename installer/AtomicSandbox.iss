; Atomic Sandbox - the per-user Windows installer (Inno Setup 6; installer spec 2026-11-12, plan Task 5).
;
; TARGET: Inno Setup 6.6.x, the newest 6.6 release (the release workflow, plan Task 6, pins that
; exact version on the runner, so every CI build compiles with the same Inno Setup). Older 6.x still compile through the
; #if Ver branches (x64compatible from 6.3.0, CreateCustomForm's new signature from 6.6.0).
;
; Built by scripts\package.ps1 (build.bat locally, the release workflow on CI) from a staging folder
; of committed files only:
;
;   ISCC.exe /DAppVersion=<root Cargo.toml version> /DStageDir=<dist\installer-stage> installer\AtomicSandbox.iss
;
; The staging folder holds AtomicSandbox.exe, README.md, LICENSE.md, installed.txt, docs\images\,
; bundled\{models,scenes,timings}\ and AtomicSandbox.ico (the setup's own icon, written by the app's
; hidden --write-icon flag; not installed). Output: dist\AtomicSandbox-Setup-<ver>.exe.
;
; Install layout (crates/app/src/paths.rs): installed.txt beside the exe puts the app in Installed
; mode; the shipped, read-only content is read from {app}\bundled\. The user's own data lives in
; %LOCALAPPDATA%\atomic-sandbox and is never touched here, except by the uninstaller's unticked
; "Also remove my ..." box (below).
;
; The in-app updater (crates/app/src/update/apply.rs, installer_command) runs this setup as
;   <setup> /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /WAITPID=<pid> /RELAUNCH="<exe> --resume-session <session>"
; - /WAITPID: wait for that process (the app, closing itself) to exit before any file is replaced
;   (InitializeSetup). A PID that is already gone means go on; after 120 s the install is aborted with
;   a log line. Setup never closes or kills any process (CloseApplications=no: the app refuses the
;   hand-off itself while another copy of this install runs).
; - /RELAUNCH: Inno's parameter reader drops the quotes, so {param:RELAUNCH} is
;   "<exe> --resume-session <session>" with no inner quotes. After a silent install it is split at
;   the first " --resume-session " and run as "<exe>" --resume-session "<session>" (RunRelaunch),
;   only when <exe> is this install's own {app}\AtomicSandbox.exe. A silent install that got past
;   the wait but failed relaunches with --update-failed added, and the app says so.
; - An installed window holds the mutex AtomicSandboxInstalledRunning (crates/app/src/instance.rs).
;   The uninstaller and an interactive setup check it (CheckForMutexes) and ask the user to close
;   the app. Not AppMutex: that would also hold up the silent update while the app is closing.
;
; Every Setup and Uninstall run writes a log (SetupLogging, UninstallLogging) in %TEMP%: a silent
; update has no window, so the log is where a failure is told.
;
; Keep this file ASCII: ISCC reads a file without a byte-order mark in the system code page.

#ifndef AppVersion
  #error Pass the version: ISCC /DAppVersion=x.y.z (the root Cargo.toml's workspace version)
#endif
#if AppVersion == ""
  #error /DAppVersion is empty
#endif
#ifndef StageDir
  #define StageDir AddBackslash(SourcePath) + "..\dist\installer-stage"
#endif
#ifndef IconFile
  #define IconFile StageDir + "\AtomicSandbox.ico"
#endif
#define AppName "Atomic Sandbox"
#define AppExe "AtomicSandbox.exe"
#define AppRepo "https://github.com/ItsKodas/atomic-sandbox"

; the staged files this script needs: fail the compile, not the install
#pragma message "StageDir = " + StageDir + ", AppVersion = " + AppVersion
#if !FileExists(StageDir + "\" + AppExe)
  #error The staging folder has no AtomicSandbox.exe: run scripts\package.ps1 first
#endif
#if !FileExists(StageDir + "\installed.txt")
  #error The staging folder has no installed.txt: run scripts\package.ps1 first
#endif
#if !FileExists(IconFile)
  #error No setup icon (IconFile): AtomicSandbox.exe --write-icon writes it, scripts\package.ps1 runs that
#endif

; the version resource takes numbers only: "0.2.0-beta.1" -> "0.2.0"
#define DashAt Pos("-", AppVersion)
#if DashAt > 0
  #define NumericVersion Copy(AppVersion, 1, DashAt - 1)
#else
  #define NumericVersion AppVersion
#endif

[Setup]
; fixed for the life of the app (generated once, 2026-11-13): Windows' Apps list and every later
; version's setup find this install by it. Never change it.
AppId={{9AFF8A1C-3706-4072-A61F-EDC763C0D4A9}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=ItsKodas
AppPublisherURL={#AppRepo}
AppSupportURL={#AppRepo}/issues
AppUpdatesURL={#AppRepo}/releases
VersionInfoVersion={#NumericVersion}
VersionInfoProductName={#AppName}
VersionInfoDescription={#AppName} Setup

; per user, no admin rights, nothing to choose but Next
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\Atomic Sandbox
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableWelcomePage=yes
ShowLanguageDialog=no
WizardStyle=modern

; never close a running app (controller ruling, plan Review Focus 4): the app refuses the update
; hand-off itself while another copy of this install runs, and waits for its own exit (/WAITPID)
CloseApplications=no
RestartApplications=no

SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
SetupLogging=yes
UninstallLogging=yes

; wgpu (Vulkan/DX12) needs 64-bit Windows 10 or newer
MinVersion=10.0
#if Ver >= EncodeVer(6, 3, 0)
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
#else
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
#endif

OutputDir={#AddBackslash(SourcePath)}..\dist
OutputBaseFilename=AtomicSandbox-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[InstallDelete]
; the shipped, read-only content is replaced whole on every install, so a model, scene or picture
; a newer version dropped doesn't linger (install folder only; the user's copies in
; %LOCALAPPDATA%\atomic-sandbox are theirs). Only in a folder that is this app's install, or new
; or empty (IsOurInstallDir): never in one with portable.txt or a Cargo.toml, whatever /DIR= says.
Type: filesandordirs; Name: "{app}\bundled"; Check: IsOurInstallDir
Type: filesandordirs; Name: "{app}\docs\images"; Check: IsOurInstallDir

[Files]
Source: "{#StageDir}\{#AppExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StageDir}\installed.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StageDir}\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StageDir}\LICENSE.md"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StageDir}\docs\images\*"; DestDir: "{app}\docs\images"; Flags: ignoreversion recursesubdirs createallsubdirs
; bundled\models (committed *.model), bundled\scenes (presets + the committed scenes\ library),
; bundled\timings (the committed timings\, possibly empty)
Source: "{#StageDir}\bundled\*"; DestDir: "{app}\bundled"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

[Icons]
; the same AppUserModelID the app sets for itself (main.rs), so a pinned shortcut and the running
; window share one taskbar button
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; AppUserModelID: "ItsKodas.AtomicSandbox"; Comment: "Atomic matter with real physics, live"

[Run]
; the Finish page's box, ticked (an interactive install only; a silent update relaunches through
; /RELAUNCH instead)
Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[Code]
const
  { Win32's SYNCHRONIZE access right and WAIT_OBJECT_0, under names of our own }
  ProcessSynchronize = $00100000;
  WaitObject0 = 0;
  { how long /WAITPID waits for the app's own exit before the install is called off }
  WaitPidTimeoutMs = 120000;
  { app/src/update/apply.rs relaunch_arg: "<exe> --resume-session <session>" }
  RelaunchSeparator = ' --resume-session ';
  { the uninstaller's "Also remove my data": the per-user root, and only these four folders in it
    (spec builder-check ruling: what the installed build created; never cache\, training\,
    current.profile or timings.txt, which dev and portable builds share) }
  DataRootName = 'atomic-sandbox';
  AttrDirectory = $10;
  AttrReparsePoint = $400;
  { OpenProcess's error for a PID that names no process (ERROR_INVALID_PARAMETER) }
  ErrorInvalidParameter = 87;
  { crates/app/src/instance.rs INSTALLED_RUNNING: held by every installed window }
  InstalledRunningMutex = 'AtomicSandboxInstalledRunning';

var
  { the setup got past the /WAITPID wait: a failed silent install may reopen what is installed }
  MayRelaunch: Boolean;
  InstallSucceeded: Boolean;
  { the uninstall form's box, and its answer }
  RemoveDataBox: TNewCheckBox;
  RemoveData: Boolean;

function OpenProcess(dwDesiredAccess: DWORD; bInheritHandle: BOOL; dwProcessId: DWORD): THandle;
  external 'OpenProcess@kernel32.dll stdcall';
function WaitForSingleObject(hHandle: THandle; dwMilliseconds: DWORD): DWORD;
  external 'WaitForSingleObject@kernel32.dll stdcall';
function CloseHandle(hObject: THandle): BOOL;
  external 'CloseHandle@kernel32.dll stdcall';

{ ---- install ---- }

{ True when an installed Atomic Sandbox window is running (its fixed-name mutex is held). }
function InstalledAppRunning(): Boolean;
begin
  Result := CheckForMutexes(InstalledRunningMutex);
end;

{ Asks the user to close every installed Atomic Sandbox window first (Retry), or to stop (Cancel,
  False). Silent with /SUPPRESSMSGBOXES: Cancel. Setup never closes the app itself. }
function AskToCloseApp(const Doing: String): Boolean;
begin
  Result := True;
  while InstalledAppRunning() do begin
    Log('Atomic Sandbox is running: asking the user to close it before ' + Doing);
    if SuppressibleMsgBox('Atomic Sandbox is running. Close every Atomic Sandbox window, then click Retry to go on with ' + Doing + '.',
         mbError, MB_RETRYCANCEL, IDCANCEL) <> IDRETRY then begin
      Log('the app was left running: ' + Doing + ' called off');
      Result := False;
      Exit;
    end;
  end;
end;

{ /WAITPID=<pid>: wait (up to WaitPidTimeoutMs) for that process to exit before anything is
  replaced. No /WAITPID: go on. A PID that names no process (OpenProcess fails with
  ERROR_INVALID_PARAMETER) has already exited: go on. Any other failure to open it (access denied:
  it is alive), still running at the cap, or a value that isn't a PID: the install is called off
  (False), nothing replaced and nothing closed. }
function WaitForPid(): Boolean;
var
  Text: String;
  Pid, Err: Integer;
  ProcessHandle: THandle;
  Waited: DWORD;
begin
  Result := True;
  Text := Trim(ExpandConstant('{param:WAITPID|}'));
  if Text = '' then
    Exit;
  Pid := StrToIntDef(Text, -1);
  if Pid <= 0 then begin
    Log(Format('/WAITPID=%s is not a process id: install called off, nothing was replaced', [Text]));
    Result := False;
    Exit;
  end;
  { only the right to wait on it: Setup never asks for, or uses, the right to end a process }
  ProcessHandle := OpenProcess(ProcessSynchronize, False, Pid);
  if ProcessHandle = 0 then begin
    Err := DLLGetLastError();
    if Err = ErrorInvalidParameter then
      Log(Format('/WAITPID=%d: that process has already exited; installing', [Pid]))
    else begin
      Log(Format('/WAITPID=%d: could not open it to wait (error %d: %s), so it may still be running: install called off, nothing was replaced and nothing was closed', [Pid, Err, SysErrorMessage(Err)]));
      Result := False;
    end;
    Exit;
  end;
  try
    Log(Format('/WAITPID=%d: waiting up to %d s for it to exit', [Pid, WaitPidTimeoutMs div 1000]));
    Waited := WaitForSingleObject(ProcessHandle, WaitPidTimeoutMs);
  finally
    CloseHandle(ProcessHandle);
  end;
  if Waited = WaitObject0 then
    Log(Format('/WAITPID=%d: it exited; installing', [Pid]))
  else begin
    Log(Format('/WAITPID=%d: still running after %d s (wait result %d): install called off, nothing was replaced and nothing was closed', [Pid, WaitPidTimeoutMs div 1000, Waited]));
    Result := False;
  end;
end;

{ An interactive setup first asks the user to close a running installed app (the silent update
  waits for its own app through /WAITPID instead, and never asks). }
function InitializeSetup(): Boolean;
begin
  MayRelaunch := False;
  if not WizardSilent() then begin
    Result := AskToCloseApp('the install');
    if not Result then
      Exit;
  end;
  Result := WaitForPid();
  MayRelaunch := Result;
end;

{ M1: the [InstallDelete] entries' check. True for this app's install folder (installed.txt in it),
  a folder that doesn't exist yet, or an empty one; never for a folder holding portable.txt (a
  portable copy) or Cargo.toml (a source checkout), whatever /DIR= pointed at. }
function IsOurInstallDir(): Boolean;
var
  Dir: String;
  Rec: TFindRec;
  Empty: Boolean;
begin
  Dir := ExpandConstant('{app}');
  if FileExists(Dir + '\portable.txt') or FileExists(Dir + '\Cargo.toml') then begin
    Log('not clearing bundled\ or docs\images\ in ' + Dir + ': it holds portable.txt or Cargo.toml');
    Result := False;
    Exit;
  end;
  if FileExists(Dir + '\installed.txt') or (not DirExists(Dir)) then begin
    Result := True;
    Exit;
  end;
  Empty := True;
  if FindFirst(Dir + '\*', Rec) then begin
    try
      repeat
        if (Rec.Name <> '.') and (Rec.Name <> '..') then
          Empty := False;
      until (not Empty) or (not FindNext(Rec));
    finally
      FindClose(Rec);
    end;
  end;
  Result := Empty;
  if not Result then
    Log('not clearing bundled\ or docs\images\ in ' + Dir + ': it is neither this app''s install folder nor empty');
end;

{ The installed exe, AtomicSandbox.exe in the app folder, expanded; empty while the app folder
  isn't known yet. }
function InstalledExe(): String;
begin
  try
    Result := ExpandFileName(ExpandConstant('{app}\{#AppExe}'));
  except
    Result := '';
  end;
end;

{ /RELAUNCH="<exe> --resume-session <session>": split at the first " --resume-session " and run
  "<exe>" --resume-session "<session>" (Exec quotes the file name itself), plus Extra. Without the
  separator the whole value is the exe, run with Extra only. Only ever this install's own exe
  (M3: compared without case to the app folder's AtomicSandbox.exe): anything else is logged,
  not run. }
procedure RunRelaunch(const Why, Extra: String);
var
  Value, Exe, Session, Params, Expected: String;
  At, Code: Integer;
begin
  Value := Trim(ExpandConstant('{param:RELAUNCH|}'));
  if Value = '' then
    Exit;
  At := Pos(RelaunchSeparator, Value);
  if At > 0 then begin
    Exe := Trim(Copy(Value, 1, At - 1));
    Session := Trim(Copy(Value, At + Length(RelaunchSeparator), Length(Value)));
    if Session <> '' then
      Params := '--resume-session "' + Session + '"'
    else
      Params := '';
  end else begin
    Exe := Value;
    Params := '';
  end;
  if Extra <> '' then begin
    if Params <> '' then
      Params := Params + ' ';
    Params := Params + Extra;
  end;
  Expected := InstalledExe();
  if (Expected = '') or (CompareText(ExpandFileName(Exe), Expected) <> 0) then begin
    Log(Format('/RELAUNCH (%s): %s is not this install''s %s; nothing started', [Why, Exe, Expected]));
    Exit;
  end;
  if not FileExists(Exe) then begin
    Log(Format('/RELAUNCH (%s): %s is not there; nothing started', [Why, Exe]));
    Exit;
  end;
  if Exec(Exe, Params, ExtractFileDir(Exe), SW_SHOWNORMAL, ewNoWait, Code) then
    Log(Format('/RELAUNCH (%s): started "%s" %s', [Why, Exe, Params]))
  else
    Log(Format('/RELAUNCH (%s): could not start "%s": %s', [Why, Exe, SysErrorMessage(Code)]));
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssDone then begin
    InstallSucceeded := True;
    if WizardSilent() then
      RunRelaunch('installed', '');
  end;
end;

{ A silent update that got past the wait but didn't finish (a full disk, a locked file): the app
  has already closed itself, so reopen what is installed now, on the saved session, rather than
  leave the user with nothing, with --update-failed so it says so (M4): Inno's rollback removes
  the files it added but doesn't restore the ones it replaced or deleted, so the install may be
  partly updated (old and new files mixed). Not after the wait was called off: the app is still
  running then. }
procedure DeinitializeSetup();
begin
  if MayRelaunch and (not InstallSucceeded) and WizardSilent() then
    RunRelaunch('the install did not finish; reopening what is installed, which may be partly updated', '--update-failed');
end;

{ ---- uninstall ---- }

procedure RemoveDataNoteClick(Sender: TObject);
begin
  RemoveDataBox.Checked := not RemoveDataBox.Checked;
end;

{ The uninstall form: what is removed, and the unticked "Also remove my ..." box. False: cancel. }
function AskUninstall(): Boolean;
var
  Form: TSetupForm;
  Intro, Note: TNewStaticText;
  UninstallButton, CancelButton: TNewButton;
begin
#if Ver >= EncodeVer(6, 6, 0)
  { 6.6.0 changed the signature: the client size is passed in unscaled (the function scales it);
    KeepSize True, so WizardSizePercent doesn't stretch the layout below }
  Form := CreateCustomForm(460, 180, True, True);
#else
  Form := CreateCustomForm();
#endif
  try
#if Ver < EncodeVer(6, 6, 0)
    Form.ClientWidth := ScaleX(460);
    Form.ClientHeight := ScaleY(180);
#endif
    Form.Caption := 'Uninstall {#AppName}';

    Intro := TNewStaticText.Create(Form);
    Intro.Parent := Form;
    Intro.AutoSize := False;
    Intro.WordWrap := True;
    Intro.Left := ScaleX(16);
    Intro.Top := ScaleY(16);
    Intro.Width := Form.ClientWidth - ScaleX(32);
    Intro.Height := ScaleY(30);
    Intro.Caption := '{#AppName} (the program in ' + ExpandConstant('{app}') + ') will be removed from this computer.';

    RemoveDataBox := TNewCheckBox.Create(Form);
    RemoveDataBox.Parent := Form;
    RemoveDataBox.Left := ScaleX(16);
    RemoveDataBox.Top := ScaleY(54);
    RemoveDataBox.Width := Form.ClientWidth - ScaleX(32);
    RemoveDataBox.Height := ScaleY(17);
    RemoveDataBox.Caption := 'Also remove my models, profiles, scenes and captures';
    RemoveDataBox.Checked := False;

    { the rest of the sentence, under the box's caption; a click on it toggles the box too }
    Note := TNewStaticText.Create(Form);
    Note.Parent := Form;
    Note.AutoSize := False;
    Note.WordWrap := True;
    Note.Left := ScaleX(16 + 18);
    Note.Top := ScaleY(74);
    Note.Width := Form.ClientWidth - ScaleX(32 + 18);
    Note.Height := ScaleY(52);
    { M6: models\ holds models\collections, the samples the app has learned from }
    Note.Caption := '(in %LOCALAPPDATA%\atomic-sandbox; models includes models\collections, your learned collections). Your pseudopotential cache and training checkpoints are kept.';
    Note.OnClick := @RemoveDataNoteClick;

    CancelButton := TNewButton.Create(Form);
    CancelButton.Parent := Form;
    CancelButton.Width := ScaleX(85);
    CancelButton.Height := ScaleY(23);
    CancelButton.Left := Form.ClientWidth - ScaleX(16 + 85);
    CancelButton.Top := Form.ClientHeight - ScaleY(14 + 23);
    CancelButton.Caption := 'Cancel';
    CancelButton.ModalResult := mrCancel;
    CancelButton.Cancel := True;

    UninstallButton := TNewButton.Create(Form);
    UninstallButton.Parent := Form;
    UninstallButton.Width := ScaleX(85);
    UninstallButton.Height := ScaleY(23);
    UninstallButton.Left := CancelButton.Left - ScaleX(8 + 85);
    UninstallButton.Top := CancelButton.Top;
    UninstallButton.Caption := 'Uninstall';
    UninstallButton.ModalResult := mrOk;
    UninstallButton.Default := True;

    Result := Form.ShowModal() = mrOk;
    RemoveData := Result and RemoveDataBox.Checked;
  finally
    Form.Free();
  end;
end;

{ A silent uninstall keeps all data (the box is never ticked without a person to tick it). A
  running installed app is asked to be closed first (M7; silent with /SUPPRESSMSGBOXES: the
  uninstall is called off rather than remove files the app holds). }
function InitializeUninstall(): Boolean;
begin
  RemoveData := False;
  Result := AskToCloseApp('the uninstall');
  if not Result then
    Exit;
  if not UninstallSilent() then
    Result := AskUninstall();
  Log(Format('Also remove my models, profiles, scenes and captures: %d', [Ord(RemoveData)]));
end;

{ Deletes what is inside Dir, depth first. A link (junction or symbolic link) is never followed
  and never removed: it and its target are left, and logged. }
procedure DeleteContentsNoLinks(const Dir: String);
var
  Rec: TFindRec;
  Path: String;
begin
  if FindFirst(Dir + '\*', Rec) then begin
    try
      repeat
        if (Rec.Name <> '.') and (Rec.Name <> '..') then begin
          Path := Dir + '\' + Rec.Name;
          if (Rec.Attributes and AttrReparsePoint) <> 0 then
            Log('kept (a link, not followed): ' + Path)
          else if (Rec.Attributes and AttrDirectory) <> 0 then begin
            DeleteContentsNoLinks(Path);
            if not RemoveDir(Path) then
              Log('could not remove the folder ' + Path);
          end else if not DeleteFile(Path) then
            Log('could not remove ' + Path);
        end;
      until not FindNext(Rec);
    finally
      FindClose(Rec);
    end;
  end;
end;

{ One of the four data folders under the root: gone whole, unless it is itself a link. }
procedure RemoveDataFolder(const Root, Name: String);
var
  Path: String;
  Rec: TFindRec;
  IsLink: Boolean;
begin
  Path := Root + '\' + Name;
  if not DirExists(Path) then
    Exit;
  IsLink := False;
  if FindFirst(Path, Rec) then begin
    IsLink := (Rec.Attributes and AttrReparsePoint) <> 0;
    FindClose(Rec);
  end;
  if IsLink then begin
    Log('kept (a link, not followed): ' + Path);
    Exit;
  end;
  DeleteContentsNoLinks(Path);
  if RemoveDir(Path) then
    Log('removed ' + Path)
  else
    Log('could not remove all of ' + Path + ' (what is left is listed above)');
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Root: String;
begin
  if (CurUninstallStep = usPostUninstall) and RemoveData then begin
    { M5, an assumption kept on purpose: the app finds its root from the LOCALAPPDATA environment
      variable (paths::user_root); the localappdata constant is the shell's known folder. They are the same
      folder unless the user has overridden the variable by hand; then this removes the four
      folders under the known folder's atomic-sandbox, not the one the app uses. }
    Root := ExpandConstant('{localappdata}') + '\' + DataRootName;
    RemoveDataFolder(Root, 'models');
    RemoveDataFolder(Root, 'profiles');
    RemoveDataFolder(Root, 'scenes');
    RemoveDataFolder(Root, 'captures');
  end;
end;
