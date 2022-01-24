FROM python:3.8.12
RUN pip install --upgrade pip
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r  requirements.txt 
EXPOSE 8501
COPY . /app
ENTRYPOINT ["streamlit","run"]
CMD ["dashboard.py"]