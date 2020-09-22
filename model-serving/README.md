# Model Serving

Using flask.

Install required packages

```bash
    pip3 install flask  flask_bootstrap  gunicorn

    # we also need tensorflow
    pip3 install tensorflow
```

To run the flask app in dev mode

```bash
    # this has a problem of printing each log entry twice! - ugh 
    $   FLASK_ENV=development FLASK_APP=app.py flask run

    # another dev mode
    $  FLASK_ENV=development python app.py
```


In production mode

```bash
    $  gunicorn --bind 0.0.0.0:5000 wsgi:app

    # using more cpu cores (workers = 2 x cores + 1)
    $  gunicorn --workers=5 --bind 0.0.0.0:8000 wsgi:app
```

Go to http://localhost:8000

For debugging

```bash
    $  docker run -u $(id -u):$(id -g)  -it --rm -v "models:/app/models" --entrypoint /bin/bash  sujee/sotu-text-generator
```