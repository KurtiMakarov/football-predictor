web: gunicorn -w ${WEB_CONCURRENCY:-2} -k gthread --threads ${GUNICORN_THREADS:-4} --timeout ${GUNICORN_TIMEOUT:-120} -b 0.0.0.0:${PORT:-5000} src.web:app
