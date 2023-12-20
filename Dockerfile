FROM nvcr.io/nvidia/pytorch:23.11-py3
RUN pip install --upgrade pip

WORKDIR /mask_rcnn
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY mask_rcnn mask_rcnn
COPY worker.py worker.py

# These should be overwritten on command line
ENV EXPERIMENT=experiments/exp_test.yaml
ENV JOBNAME=TEST

CMD python worker.py
