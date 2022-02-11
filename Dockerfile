FROM python:3.8.12
RUN pip install --upgrade pip
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y vim
RUN pip3 install -r  requirements.txt 
ENV PORT=4000
COPY . /app
CMD streamlit run dashboard.py --server.port=${PORT}  --browser.serverAddress="0.0.0.0"