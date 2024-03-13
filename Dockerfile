FROM python:3.9
WORKDIR /app
ADD . .
RUN pip install -r requirements.txt
CMD [ "streamlit", "run", "parkinsonsSpyder.py" ]