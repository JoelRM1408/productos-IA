FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY streamlit_app.py ./streamlit_app.py
COPY loader_modelos.py ./loader_modelos.py 
COPY supabase_client.py ./supabase_client.py 
COPY artefactos ./artefactos
COPY styles ./styles
COPY img ./img
ENV ART_DIR=artefactos
EXPOSE 8501
CMD ["streamlit","run","streamlit_app.py","--server.port=8501","--server.address=0.0.0.0"]
