# computervisionCv2

Steps Required to Run Car Detection GUI

1. Install all the requirements to run Flask based web server
   pip3 install -r requirements.txt

2. Download weights file the drive link 
   and place it in yolo-stanfordcar-data
   (As weights file in 240 mb github is not alllowing to host it)

3. Start the Flask web server by running
   python3 detect-flask.py

   Go to the URL : http://localhost:8090

4. ![image](https://user-images.githubusercontent.com/11522867/164389365-222fbafe-d95d-4bb7-882c-234238a98964.png)
    From the screenshot press Choose file and select any image from test folder
    For images selected from test folder its class will be displayed but for new images class will be shown as N/A
    Press Predict when image is selected
    
    Below is the result of predicted image . 
    Predicted output consists of 
    1. Image with bounding box
    2. Predicted class
    3. Confidence Score
    
    <img width="558" alt="image" src="https://user-images.githubusercontent.com/11522867/164390065-8958d162-83f6-43ce-befc-e7bb9c9afc87.png">
