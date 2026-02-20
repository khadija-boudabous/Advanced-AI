# ModIA Advanced AI Practical Programming Repository

This repository contains code for practical programming sessions of the Advanced AI course. Each directory corresponds to a session. It will be updated along the semester with code for each session, including exercises and sample solutions.

## Repository Setup

First fork this repository to your own GitHub account, then clone it locally

```bash
git clone your_forked_URL
```

Then add this repo as a remote to pull updates

```bash
git remote add upstream https://github.com/paulnovello/Advanced-AI
```

To update your forked repo with the latest changes from this original repo, run:

```bash
git fetch upstream
git merge upstream/main
```



## Environment Setup

### 1. Install uv

uv is a fast Python package installer and resolver. Install it using one of the following methods:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Using pip:**
```bash
pip install uv
```

### 2. Create env

```bash
uv env ./aai_env
```


### 2. Install Dependencies

Install the required dependencies using uv:

```bash
uv sync --group student
```

### 3. Activate env

Activate the virtual environment created by uv:

```bash
source ./aai_env/bin/activate  # macOS/Linux
```
or
```bash
.\aai_env\Scripts\activate    # Windows
```

or use uv's run command to execute scripts within the environment without activating it:

```bash
uv run python # my_script.py
```