#Uses py 3.9.23 version 
FROM python:3.9.23-slim

#Sets the working directory in the container
WORKDIR /app

#copy the requrements for the container
COPY requirements.txt .

#install dependencies 
RUN pip install --no-cache-dir -r requirements.txt

#copy the appication code into the container 
COPY . .

#expose
EXPOSE 8000


#set the command to run the FastApi Appication
CMD [ "uvicorn","main:app","--host","0.0.0.0","--port","8000" ]