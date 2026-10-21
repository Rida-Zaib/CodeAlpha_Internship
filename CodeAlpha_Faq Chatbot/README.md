TechNest AI Support — FAQ Chatbot
A customer-support chatbot that understands questions however they are worded: short, long, casual, with typos, or several questions in one message.

Smart mode (needs one free API key): an LLM reads the customer's message together with the entire FAQ list and answers from it. It copes with any phrasing and remembers the last few messages, so follow-ups like "and for international?" work. It is told to answer only from the FAQ and to say so when the FAQ doesn't cover something.
Local matcher (no key, no internet): classic NLP (NLTK + TF-IDF / cosine similarity) over the FAQs and their alternate phrasings. Instant, free, and used automatically whenever Smart mode is off or unreachable.
faq-chatbot/
├── faqs.csv              → FAQ dataset (category, question, answer)
├── preprocess.py          → text cleaning: tokenize, remove stopwords, lemmatize
├── matcher.py               → the matching engine (see "How it works")
├── app.py                     → Flask server: serves the UI + /ask, /feedback, /health
├── chatbot_cli.py                → terminal chatbot (same engine, no browser needed)
├── gap_report.py                   → summarizes which questions the bot couldn't answer
├── templates/index.html              → the chat UI
├── tests/test_matcher.py               → sanity tests for the matching logic
├── data/                                 → created at runtime: unanswered.csv, feedback.csv
└── requirements.txt
1. Setup
cd faq-chatbot
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
Then download the NLTK data used for preprocessing (one-time):

python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
If you skip this, the bot still runs — preprocess.py falls back to a built-in stopword list instead of crashing — but matching is a bit sharper once the real NLTK data is in place (proper lemmatization: "returning" / "returns" / "returned" all collapse to the same word).

When you start the app, the terminal prints one line saying whether NLTK is really being used ("Text preprocessing: NLTK active ..."), or exactly what is missing and the command to fix it. GET /health shows the same under nlp.

2. Run it
Web chat UI (recommended):

python app.py
Open http://127.0.0.1:5000.

Terminal version:

python chatbot_cli.py
Run the tests (optional, needs pip install pytest):

pytest tests/ -v
3. How it works
faqs.csv — 25 Q&A pairs for a fictional electronics store, each tagged with a category (Shipping, Returns, Payments, Warranty, Account, Support).
preprocess.py — lowercases, strips punctuation, tokenizes, removes stopwords, lemmatizes.
matcher.py is the local engine. For every message it:
indexes each FAQ under its question plus the alternate phrasings in the alternates column of faqs.csv (including Roman Urdu), and also searches the answer text;
splits long messages into clauses so filler words don't drown the real question, and answers every part of a multi-part question;
expands synonyms ("money back" becomes "refund"), recognises sentence shapes ("when will my stuff show up" means delivery time) and corrects typos;
says so honestly when nothing matches, offers the closest questions it can answer, and logs the question to data/unanswered.csv. Run python gap_report.py to see which FAQs or alternates to add next.
app.py exposes:
POST /ask with {message, history, mode} returning {answer, answered, suggestions, ...}. Greetings and thanks are answered instantly without using any AI quota.
POST /feedback logs the thumbs up/down to data/feedback.csv, so you can see which answers were actually wrong.
GET /health returns status, number of FAQs, and which AI providers are configured.
4. Smart mode — AI-phrased answers (optional)
By default the bot still answers purely from TF-IDF matching (see above) — that part needs no setup and never calls the internet. On top of that, Smart mode sends the FAQ(s) that TF-IDF already retrieved to a free LLM API, which rephrases the answer more naturally. The LLM is never allowed to answer from its own knowledge — only from the FAQ context it's given — so it can't invent policies or numbers that aren't in faqs.csv. This is retrieval-augmented generation (RAG), not fine-tuning: nothing about the model itself is "trained" on this data, we just feed it the right paragraph at request time.

Setup (optional, the bot works fine without this):

Copy .env.example to .env.
Get a free API key from one or more of these (no credit card needed for any of them):
Groq — console.groq.com/keys → GROQ_API_KEY
Google Gemini — aistudio.google.com/apikey → GEMINI_API_KEY
OpenRouter — openrouter.ai/keys → OPENROUTER_API_KEY
Hugging Face (optional, least reliable) — huggingface.co/settings/tokens → HF_TOKEN. Its free allowance for the "Inference Providers" router is small, and many accounts get HTTP 402 (payment required) or HTTP 400 (model not served) straight away. The app detects that, says so once in the terminal, and stops trying Hugging Face for 30 minutes (or, for a rejected model, until restart). So it never slows anything down, but don't rely on it: Groq is the one to set up first.
Paste whichever key(s) you got into .env. One is enough; set up two to four for automatic fallback if one is down, slow, or rate-limited.
Restart python app.py.
Models get retired, and the app copes. Free-tier models are switched off every few months (Groq's llama-3.3-70b-versatile in August 2026, Gemini 2.0 in June 2026). A hard-coded model name would then fail forever and the bot would quietly fall back to plain FAQ text. So for Groq, Gemini and OpenRouter the app tries its current models in order, remembers any the provider says are gone, and if all of them are gone it asks the provider which models it offers today and uses one of those. You can pin your own with GROQ_MODEL= etc. in .env.

If answers look like copied FAQ text instead of natural replies, the AI is not working and the bot is using its basic matcher. The header then says "AI offline, using basic answers". Run:

python check_ai.py
It makes one real request per key and prints the exact reason (rejected key, rate limit, no internet, ...) in plain words. GET /health also shows it under ai_errors, and the terminal logs one line per failure.

How the fallback chain works — racing, not one-at-a-time: on every Smart-mode request, app.py calls every configured provider at the same time, each on its own thread — Groq, Gemini and OpenRouter each individually, and Hugging Face (which internally tries its own list of 6 models) as a fourth. Whichever one answers first wins; the rest are left to finish in the background and are simply discarded. This is much faster than the obvious approach of trying them one at a time and waiting out each failure/timeout before moving to the next — in testing, a race with one slow (5s) and one fast (0.3s) provider returned in 0.3s, not 5s. If none are configured, or literally none of them respond within 20 seconds, it silently falls back to the instant TF-IDF answer — the chatbot never breaks because of this. The chat UI has a ✨ Smart / ⚡ Fast toggle in the header so you (or anyone testing it) can compare both modes directly, and every answer shows which engine actually won the race.

Note: Smart mode only calls the LLM when TF-IDF already found a relevant FAQ. Gibberish or unrelated questions skip the AI call entirely and go straight to the honest "I don't know" fallback — this also means Smart mode never burns API quota on junk input.

Hugging Face: what to expect
Hugging Face is the optional extra, not the backbone. Details:

Six models, tried in order (DEFAULT_HF_MODELS; override with HF_MODELS=model/one,model/two,... in .env).
Dead ends are remembered. HTTP 402 (no credits) or 401/403 (bad token) pauses Hugging Face for 30 minutes; HTTP 400/404/422 (model not served) drops just that model until restart. Each is logged once in plain words, and GET /health stops listing Hugging Face, so the Smart/Fast switch doesn't appear because of a dead token.
Startup warm-up. With a working token, a background thread pings each model once at startup so the first visitor doesn't wait for a cold model. GET /health → hf_models_warmed shows which ones answered.
Persistent answer cache (ai_cache.py). Smart-mode answers are cached on disk (data/ai_cache.json), keyed by the customer's exact question (ignoring case and punctuation). Asking the same stand-alone question again is instant, with no API call. Follow-up questions are not cached, because their meaning depends on the conversation. Delete data/ai_cache.json (or call ai_cache.clear()) after editing faqs.csv, so old answers don't linger.
5. What the UI shows
A deliberately plain chat: the answer, and nothing else cluttering it.

Starter chips with a few example questions.
Thumbs up / down under every real answer. They feed data/feedback.csv, which is how wrong answers get found and fixed.
"You could try asking:" chips, only when the bot couldn't answer, offering the closest questions it does know.
Smart / Fast switch in the header. It only appears when an AI key is configured (without one it would do nothing).
New conversation button (clears the chat and its memory).
6. Using your own FAQs
Edit faqs.csv — keep the category,question,answer,alternates header (alternates is optional: other ways people ask the same thing, separated by |; a few of them make the no-key local matcher much more forgiving). If you add new wording gaps (e.g. your customers say "cancel my purchase" but your FAQ says "cancel order"), add that pair to the SYNONYMS dict at the top of matcher.py. No other code changes needed — both chatbot_cli.py and app.py rebuild the TF-IDF index from the CSV on startup.

7. Putting the project on GitHub
Do this first: hosting platforms deploy from a GitHub repository.

Install Git from git-scm.com (check with git --version). Once, tell it who you are:
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
On github.com click New repository. Give it a name (for example CodeAlpha_FAQ_Chatbot), choose Public or Private, and leave Add a README / .gitignore / license all unticked (this project already has them).
In a terminal inside the project folder:
git init
git add .
git status
Read the list git status prints. .env must NOT be in it (it holds your API keys; .gitignore keeps it out). .env.example should be there.
Then:
git commit -m "FAQ chatbot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
The first push opens a browser window to sign in to GitHub (GitHub no longer accepts your password in the terminal).
Refresh the repo page on github.com and check the files are there and .env is not. If a key was ever committed by mistake, delete that key at the provider and create a new one; removing the file later does not remove it from the history.
Later changes: git add ., git commit -m "what changed", git push.

8. Deploying it (free)
Backend and frontend deploy together, as one service. app.py serves both the chat page and the /ask, /health, /feedback APIs, so there is one app and one live URL, and no CORS setup.

Why not Vercel: it runs Python as short-lived serverless functions, where nothing written to disk survives between requests. That breaks the on-disk answer cache and the startup warm-up. Use it for static sites; not for this.

Recommended: Render. It runs your app as a normal, always-on process. What the free plan gives you (as of Oct 2026, check render.com/pricing for changes): 512 MB RAM, a small share of a CPU, 750 free hours a month, no persistent disk. It may ask for a payment method when you create the service (sources disagree); free usage is not billed.

Push the project to GitHub (section 7).
render.com -> sign up with GitHub -> New + -> Web Service -> Build and deploy from a Git repository -> pick your repo (allow Render access to it if asked).
Fill in:
Name: anything (it becomes part of the URL)
Region: the nearest one (Singapore for Pakistan)
Branch: main Root Directory: leave empty Runtime: Python 3
Build Command:
pip install -r requirements.txt && python setup_nltk.py
Start Command (Render does not read the Procfile, so type it here):
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 90
Instance Type: Free
Environment Variables -> add the same names as in your .env, e.g. GROQ_API_KEY = your key. (Skip HF_TOKEN. The values are typed here, never committed to GitHub.)
Create Web Service. Watch the Logs. A good run ends with Loaded 25 FAQs, Text preprocessing: NLTK active ... and Your service is live. Your URL looks like https://your-name.onrender.com.
Open https://your-name.onrender.com/health. ai_providers_configured should list groq. Then chat on the main page.
Every git push to main redeploys automatically.
What to expect on the free plan: after 15 minutes without visitors it sleeps and the next visit takes about a minute to wake it. Files written while running (data/ai_cache.json, feedback and unanswered logs) are lost on every restart or sleep; that is only a cache and logs, nothing breaks. Anyone with the link can use your AI key's free quota, so share it with care.

If something goes wrong:

Build fails while installing scikit-learn/numpy: add the environment variable PYTHON_VERSION = 3.12.8 and redeploy.
"No open ports detected" / site won't open: the Start Command must contain --bind 0.0.0.0:$PORT exactly.
Log says "NLTK data missing": the build command is missing && python setup_nltk.py.
Header says "AI offline": open /health and look at ai_errors, or check the key's name and value under Environment.
"Out of memory": keep --workers 1.
No card? Use PythonAnywhere instead (free, no payment method)
Render may ask for a card. PythonAnywhere's free "Beginner" account does not. Its free plan only lets your app reach an allowlist of internet sites, and the allowlist (checked Oct 2026 at pythonanywhere.com/whitelist) includes api.groq.com and *.googleapis.com (Gemini), so Smart mode works. I could not confirm OpenRouter is on it, so use a Groq or Gemini key. Free limits: 512 MB disk, 1 web app with 1 worker, 100 CPU-seconds a day for consoles and tasks (web requests do not count, per their docs), and the web app expires after 1 month unless you click the extend button (step 8).

Create a free Beginner account at pythonanywhere.com (the username you pick becomes your address: USERNAME.pythonanywhere.com).
Consoles tab -> start a Bash console.
Get the code (the GitHub repo must be Public, which is safe because .env is never uploaded; or upload the zip in the Files tab and unzip it):
git clone https://github.com/YOUR_USERNAME/CodeAlpha_FAQ_Chatbot.git
Make a virtualenv and install (the --system-site-packages part reuses packages PythonAnywhere already has, which saves disk space):
mkvirtualenv --python=python3.12 --system-site-packages faqbot
cd ~/CodeAlpha_FAQ_Chatbot
pip install -r requirements.txt
python setup_nltk.py
If python3.12 is not found, try python3.11 or python3.10, and use the same version in step 6.
Create your keys file (it is not on GitHub): nano .env, type GROQ_API_KEY=your_key, then Ctrl+O, Enter, Ctrl+X. Test it: python check_ai.py should show OK for groq.
Web tab -> Add a new web app -> Manual configuration (not the Flask option) -> pick the Python version you used in step 4.
On the Web tab set:
Source code and Working directory: /home/USERNAME/CodeAlpha_FAQ_Chatbot
Virtualenv: /home/USERNAME/.virtualenvs/faqbot
Click the WSGI configuration file link, delete everything in it, paste:
import os, sys
path = "/home/USERNAME/CodeAlpha_FAQ_Chatbot"
if path not in sys.path:
    sys.path.insert(0, path)
os.chdir(path)   # the app reads faqs.csv and writes data/ relative to this folder
from app import app as application
(replace USERNAME everywhere), then Save.
Click the green Reload button, then open https://USERNAME.pythonanywhere.com/health. Do this every month: Web tab -> the button that extends the web app's life by a month. To update later: cd ~/CodeAlpha_FAQ_Chatbot && git pull, then Reload.
If something fails: Web tab -> Error log. "Disk quota exceeded": delete the virtualenv (rmvirtualenv faqbot) and redo step 4 with --system-site-packages. Smart mode works in the console but the site says "AI offline": open /health and read ai_errors.

9. Known limitation
TF-IDF + keyword overlap matches on words, not deep meaning. The synonym map and typo correction close the most common gaps, but a truly novel phrasing of a question may still miss. For stronger semantic matching you'd move to sentence embeddings (e.g. sentence-transformers), which is a heavier dependency than this project needs — gap_report.py is there so you can see exactly which real questions are being missed and patch SYNONYMS or faqs.csv accordingly.