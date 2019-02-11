FROM jupyter/scipy-notebook:83ed2c63671f
USER root
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
CMD start-notebook.sh --NotebookApp.token=''