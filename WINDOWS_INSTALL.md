# Running EXOZIPPy on Windows (via WSL2)

EXOZIPPy does not run natively on Windows, but it runs on Windows 10
(build 19041, May 2020, or newer) and Windows 11 through WSL2 (Windows
Subsystem for Linux), which runs a full Linux kernel inside Windows.
This document sets up that Linux environment. Once it is done, you
follow the ordinary Linux instructions in [`README.md`](README.md).

(Why not natively: [`docs/windows-native.md`](docs/windows-native.md)
lists the blockers.)

## Step 0 -- Check your Windows version

WSL2 needs **build 19041 or newer** (Windows 10 version 2004, or any
Windows 11). Check with `winver`, or from PowerShell:

```powershell
[System.Environment]::OSVersion.Version
```

On anything older, install Linux directly instead; there is no way to
run EXOZIPPy on it.

## Step 1 -- Install WSL2 + Ubuntu

Open an **elevated** PowerShell (type "PowerShell" in the Start menu
search, then click "Run as administrator") and run:

```powershell
wsl --install
```

This installs WSL2 and, by default, Ubuntu. **Then REBOOT.** The
required Windows components only activate after a reboot.

If, after the reboot, there is no "Ubuntu" app in the Start menu, open an
elevated PowerShell again and install it explicitly:

```powershell
wsl --install -d Ubuntu
```

Open Ubuntu (Start menu -> Ubuntu). On first launch, it asks for a UNIX
username and password. These belong to the Linux environment only and
are unrelated to your Windows login. They can match it or not, but pick
something you will remember: `sudo` asks for this password.

Nothing else needs installing inside Ubuntu: the README's conda setup
supplies the compiler, and Ubuntu already has `git` and `curl`.

## Step 2 -- Give WSL more memory (optional)

WSL2 gets **50% of the computer's RAM** by default. That is usually
enough for a fit, but may not be on a machine with 8 GB of RAM or less,
or for a large fit. If you want to raise it, continue; otherwise skip to
Step 3.

On the **Windows** side (not inside Ubuntu), create a text file named
`.wslconfig` in your Windows user folder, i.e. `C:\Users\<you>\.wslconfig`
(`/mnt/c/Users/<you>/.wslconfig` as seen from Ubuntu). Paste the
following into it and save:

```ini
[wsl2]
memory=12GB
processors=8
swap=4GB
```

Size `memory` to leave Windows a few GB, and `processors` to at most the
number of cores the machine has. `memory` is a ceiling, not a
reservation -- WSL gives unused memory back -- so setting it too high
only costs Windows some swapping.

Then, from PowerShell (it need not be elevated):

```powershell
wsl --shutdown
```

and reopen Ubuntu. **This kills every running Ubuntu shell and anything
running in it**, so do not do it in the middle of a fit. Check the new
total inside Ubuntu with `free -h`.

## Step 3 -- Continue with the Linux instructions

From here on everything happens inside the Ubuntu shell, exactly as on
any Linux machine: follow "Installing" in [`README.md`](README.md),
starting at "Install Miniforge".

Two WSL-specific habits:

- **Keep your work on the Linux filesystem** (anywhere under `~/`), not
  under `/mnt/c/`. Files under `/mnt/c/` are your Windows drive, and
  reading them from Linux is far slower.
- **Seeing Linux files from Windows:** in File Explorer, click "Linux"
  at the bottom of the left-hand pane, then "Ubuntu", and navigate to
  `home/<user>/...`. Or, from inside Ubuntu, run `explorer.exe .` to
  open the current directory in File Explorer; that is the easiest way
  to open a fit's PDF plots with your usual Windows viewer.

If you will be developing EXOZIPPy rather than only running fits, [`CONTRIBUTING.md`](CONTRIBUTING.md) has a section on
the extra WSL traps that affect developers.
