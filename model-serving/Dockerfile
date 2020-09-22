FROM tensorflow/tensorflow:latest

# upgrade pip3
RUN /usr/bin/python3 -m pip install --upgrade pip
#RUN pip3 install --upgrade pip3

RUN pip3 install numpy pandas 
RUN pip3 install scikit-learn 

RUN pip3 install flask  flask_bootstrap
RUN pip3 install gunicorn   
#RUN pip3 install google-cloud-logging 
#RUN pip3 install google-cloud-monitoring 
  
# copy app assets
ARG APP_DIR=/app
RUN mkdir ${APP_DIR}
COPY app.py ${APP_DIR}/app.py
COPY wsgi.py ${APP_DIR}/wsgi.py
COPY templates/ ${APP_DIR}/templates/
COPY static/ ${APP_DIR}/static/
COPY models/ ${APP_DIR}/models/

WORKDIR  ${APP_DIR}
ENTRYPOINT ["gunicorn", "--workers", "2", "--bind", "0.0.0.0", "wsgi:app"]