# pull python base image
FROM python:3.10

# add requirements file & trained model
ADD requirements.txt .
ADD *.pkl .

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# add application file
ADD app.py .

# expose port where your application will be running
EXPOSE 7860

# start application
CMD ["python", "app.py"]