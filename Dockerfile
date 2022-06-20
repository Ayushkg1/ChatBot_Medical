FROM python:3.7

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install gunicorn

WORKDIR /app

ADD . .

# For Local Machine
CMD python manage.py runserver 0.0.0.0:$PORT

# For Deployment
# CMD gunicorn base.wsgi:application --bind 0.0.0.0:$PORT
# EXPOSE 8000