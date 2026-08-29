# Running Sync.AI on your own computer

The hosted version of Sync.AI runs on a shared, resource-limited server —
which is why only one person can really process a video at a time, and
why it can slow down or even restart if too many people use it at once.

Running it on your own computer fixes that completely: it's just you and
your machine, no sharing, no limits, and no waiting on anyone else. It
also means your session history actually sticks around, instead of
disappearing if the hosted app goes to sleep.

This guide assumes no coding experience. It'll take about 15–20 minutes
the first time, mostly spent waiting for things to download.

---

## What you'll need

- A Windows, Mac, or Linux computer
- About 3 GB of free disk space (for the AI models and supporting software)
- An internet connection for the one-time setup (after that, it runs
  fully offline)

---

## Step 1 — Install Python

Skip this step if you already have Python 3.10 or newer installed.

**Windows:**
1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Click the big "Download Python" button
3. Run the installer. **Important:** on the first screen, check the box
   that says **"Add python.exe to PATH"** before clicking Install
4. Once it finishes, open the Start menu, type `cmd`, and press Enter to
   open Command Prompt

**Mac:**
1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Download the macOS installer and run it, following the prompts
3. Open the **Terminal** app (search for it with Spotlight — Cmd+Space,
   then type "Terminal")

To check it worked, type this and press Enter (Windows: `python
--version`, Mac: `python3 --version`). You should see something like
`Python 3.11.x`. If you instead see an error, restart your computer and
try again — this is almost always a PATH issue that a restart fixes.

---

## Step 2 — Get the app files

You'll need all of these files together in one folder:

- `app.py`
- `tracker_core.py`
- `scorer.py`
- `session_store.py`
- `issue_reports.py`
- `requirements.txt`
- `camera_angle_guide.png`
- `framing_example.png`

If these live in a GitHub repository, the easiest way is to download the
whole repository as a ZIP (green "Code" button on GitHub → "Download
ZIP") and unzip it somewhere easy to find, like your Desktop.

You do **not** need `packages.txt` — that file is only used by the
hosted (Streamlit Community Cloud) version to install Linux system
packages, which your own Windows or Mac setup won't need.

---

## Step 3 — Open a terminal in that folder

**Windows:** Open the unzipped folder in File Explorer, click in the
address bar at the top, type `cmd`, and press Enter — this opens Command
Prompt already pointed at that folder.

**Mac:** Open Terminal, type `cd ` (with a trailing space), then drag the
unzipped folder from Finder into the Terminal window and press Enter.

---

## Step 4 — Set up a clean Python environment

This keeps the app's requirements separate from anything else on your
computer. Copy-paste each line one at a time and press Enter, waiting
for each to finish.

**Windows:**
```
python -m venv venv
venv\Scripts\activate
```

**Mac:**
```
python3 -m venv venv
source venv/bin/activate
```

You'll know it worked if you see `(venv)` appear at the start of your
terminal line.

---

## Step 5 — Install everything the app needs

Still in the same terminal:
```
pip install -r requirements.txt
```

This downloads Streamlit, the AI pose-tracking library, and everything
else — it can take several minutes depending on your internet speed.
This is a one-time step; you won't need to do it again unless you delete
the `venv` folder.

---

## Step 6 — Run it

```
streamlit run app.py
```

Your browser should open automatically to the app. If it doesn't, the
terminal will print a web address (something like
`http://localhost:8501`) — copy that into your browser.

**The first video you process will still take about a minute to load the
pose model** — that part's downloading and initializing the AI model
itself, which happens once per computer, not once per person. Every
video after that, on that same computer, will be fast.

---

## Using it day to day

Once it's set up, running it again is just:
1. Open a terminal in the app's folder
2. Activate the environment (Step 4's second line only — `venv\Scripts\activate`
   on Windows, or `source venv/bin/activate` on Mac)
3. Run `streamlit run app.py`

Closing the terminal window stops the app. Reopening it and running
`streamlit run app.py` again picks up right where the History page left
off — your session data lives in a `sessions` folder that gets created
right next to `app.py`.

---

## If something goes wrong

**"streamlit: command not found" / "'streamlit' is not recognized"**
The virtual environment isn't active. Repeat Step 4's activate line — it
needs to be run every time you open a new terminal.

**Something about `libGL.so` or OpenCV on Linux**
Debian/Ubuntu users may need `sudo apt-get install libgl1
libglib2.0-0t64` first — this is the one thing `packages.txt` is for.
Not needed on Windows or Mac.

**The pose model download step seems stuck**
It's a large file and can take a few minutes on a slower connection —
give it a couple of minutes before assuming it's frozen. If the app
crashes outright, closing and re-running `streamlit run app.py` is
usually enough; the download resumes rather than starting over.

**Videos process, but much slower than expected**
Processing speed depends on your own computer's CPU — a modern laptop
or desktop should still be noticeably faster than the shared hosted
version, but an older or lower-power machine may take a while. This is a
hardware factor, not something in the app itself.
