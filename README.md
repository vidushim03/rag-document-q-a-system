# RAG PDF Project

This app now uses a persistent Qdrant-backed knowledge base instead of temporary upload-only indexing.

## What changed

- Documents are stored in Qdrant and remain available across sessions.
- Re-uploading the same filename replaces the older version in the active embedding mode.
- You can query the saved knowledge base from both the Streamlit app and the CLI.
- Local Qdrant mode works without running a separate Qdrant server.

## Windows setup

1. Install Python 3.11 or newer and enable `Add python.exe to PATH`.
2. Open PowerShell in this folder.
3. Run:

```powershell
.\setup_windows.ps1
```

4. Copy `.env.example` to `.env`.
5. Add your `GROQ_API_KEY`.
6. Start the app:

```powershell
.\run_app.ps1
```

## Qdrant configuration

By default the app uses local on-disk Qdrant storage under `knowledge_base/`.

If you want to connect to a remote Qdrant instance, set these in `.env`:

```env
QDRANT_URL=https://your-qdrant-instance
QDRANT_API_KEY=your_api_key
```

If `QDRANT_URL` is empty, the app uses local persistent storage.

## OCR configuration

The app now tries local OCR first with Tesseract and Poppler, then falls back to Groq OCR if local extraction is unavailable or too weak.

Add these to `.env` if you installed them manually on Windows:

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
POPPLER_PATH=C:\path\to\poppler\Library\bin
```

If those values are blank, the app will still work, but scanned PDFs and images will rely more heavily on Groq OCR.

## Optional packages

The core app uses TF-IDF embeddings for the most compatible first run.

- `requirements.txt`: core app dependencies, including Qdrant
- `requirements-optional.txt`: optional transformer embedding packages

Install optional transformer extras with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

## CLI

To index PDFs from the `data/` folder into Qdrant and query them from the terminal:

```powershell
python main.py
```

## Notes

- The checked-in `.venv` in this repo was copied from another operating system and should not be reused on Windows.
- Local Qdrant mode is the best fit for getting started quickly on this project.
- If package compatibility becomes an issue on Python 3.14, use Python 3.11 or 3.12 for the smoothest setup.

## Deploy to Streamlit Community Cloud

The app is ready to deploy from this repository. Local on-disk Qdrant does not work on the cloud (the filesystem is ephemeral), so you must point the app at a Qdrant Cloud cluster.

1. Create a free Qdrant Cloud cluster at <https://cloud.qdrant.io> and note the cluster URL and API key.
2. Push this repository to GitHub (see below).
3. On <https://share.streamlit.io>, click **Create app**, connect your GitHub repo, and set the main file to `app.py`.
4. In **Advanced settings**:
   - Set **Python version** to `3.12` (Community Cloud ignores `runtime.txt`).
   - Paste the contents of `.streamlit/secrets.toml.example` into **Secrets**, filling in your real `GROQ_API_KEY`, `TAVILY_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY`. Secrets are injected as environment variables, which is how the app reads them.
5. Deploy. `packages.txt` installs `tesseract-ocr` and `poppler-utils` for OCR support.

Create the GitHub repo and push from PowerShell:

```powershell
git init
git add .
git commit -m "Initial commit"
gh repo create rag-document-q-a-system --public --source . --push
```

If you do not use the GitHub CLI, create an empty repo on github.com, then run:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/rag-document-q-a-system.git
git branch -M main
git push -u origin main
```

