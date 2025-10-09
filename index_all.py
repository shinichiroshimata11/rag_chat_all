Your app is in the oven

[     UTC     ] Logs for ragchatall-5iwbkeysi9a5of94n34rps.streamlit.app/

────────────────────────────────────────────────────────────────────────────────────────

[05:56:52] 🖥 Provisioning machine...

[05:56:52] 🎛 Preparing system...

[05:56:52] ⛓ Spinning up manager process...

[05:52:51] 🚀 Starting up repository: 'rag_chat_all', branch: 'main', main module: 'chat_app_all.py'

[05:52:51] 🐙 Cloning repository...

[05:52:52] 🐙 Cloning into '/mount/src/rag_chat_all'...

[05:52:52] 🐙 Cloned repository!

[05:52:52] 🐙 Pulling code changes from Github...

[05:52:53] 📦 Processing dependencies...


──────────────────────────────────────── uv ───────────────────────────────────────────


Using uv pip install.

Using Python 3.13.8 environment at /home/adminuser/venv

  × No solution found when resolving dependencies:

  ╰─▶ Because langchain-core>=0.2.35,<=0.2.43 depends on pydantic>=2.7.4 and

      only the following versions of langchain-core are available:

          langchain-core<=0.2.35

          langchain-core==0.2.36

          langchain-core==0.2.37

          langchain-core==0.2.38

          langchain-core==0.2.39

          langchain-core==0.2.40

          langchain-core==0.2.41

          langchain-core==0.2.42

          langchain-core==0.2.43

          langchain-core>0.3.0

      we can conclude that langchain-core>=0.2.35,<0.3.0 depends on

      pydantic>=2.7.4.

      And because only the following versions of pydantic are available:

          pydantic<=2.7.4

          pydantic>=2.8.0,<=2.8.2

          pydantic>=2.9.0,<=2.9.2

          pydantic>=2.10.0,<=2.10.6

          pydantic>=2.11.0,<=2.11.10

          pydantic>=2.12.0

      and langchain-openai==0.1.23 depends on langchain-core>=0.2.35,<0.3.0,

      we can conclude that langchain-openai==0.1.23 depends on one of:

          pydantic==2.7.4

          pydantic>=2.8.0,<=2.8.2

          pydantic>=2.9.0,<=2.9.2

          pydantic>=2.10.0,<=2.10.6

          pydantic>=2.11.0,<=2.11.10

          pydantic>=2.12.0


      And because you require langchain-openai==0.1.23 and pydantic==1.10.13,

      we can conclude that your requirements are unsatisfiable.

Checking if Streamlit is installed

Installing rich for an improved exception logging

Using uv pip install.

Using Python 3.13.8 environment at /home/adminuser/venv

Resolved 4 packages in 89ms

Prepared 4 packages in 115ms

Installed 4 packages in 13ms

 + markdown-it-py==4.0.0

 + mdurl==[2025-10-09 05:52:55.343910] 0.1.2

 + pygments==2.19.2

 + rich==14.1.0


────────────────────────────────────────────────────────────────────────────────────────



──────────────────────────────────────── pip ───────────────────────────────────────────


Using standard pip install.

Collecting streamlit==1.38.0 (from -r /mount/src/rag_chat_all/requirements.txt (line 1))

  Downloading streamlit-1.38.0-py2.py3-none-any.whl.metadata (8.5 kB)

Collecting pandas==2.2.2 (from -r /mount/src/rag_chat_all/requirements.txt (line 2))

  Downloading pandas-2.2.2.tar.gz (4.4 MB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.4/4.4 MB 61.5 MB/s eta 0:00:00[2025-10-09 05:52:56.605963] 

  Installing build dependencies: started

  Installing build dependencies: finished with status 'done'

  Getting requirements to build wheel: started

  Getting requirements to build wheel: finished with status 'done'

  Installing backend dependencies: started

  Installing backend dependencies: finished with status 'done'

  Preparing metadata (pyproject.toml): started

main
shinichiroshimata11/rag_chat_all/main/chat_app_all.py


