# computervisionCv2

## Steps for Generating EDA GUI.

Run the IPYNB file EDA-GUI-Autoplotter_SK.ipynb. At the end it will generate the link for Auomation GUI of EDA.

Steps implemented in Notebook.
1. Install Autoplotter
2. Import GUI Libraries Like Jupier_Dash, Dash
3. Install "dash-bootstrap-components<1"
4. Import Run_app from Autoplotter
5. Manipulate the Data for giving input for Autoplotter
5. Click the link Generated by Autoplotter app


## Steps for Runninng training Automation

1. Execute the file tkinter_test.py
2. In Folder text box enter name of folder containing training images.
3. Press PreProcessYolov3 button
4. Press Train after Input Preprocessing Complete is shown in result box
5. Training output can found in same directory myoutput_a.txt file

<img width="813" alt="image" src="https://user-images.githubusercontent.com/11522867/164959708-9e584dcb-b632-477a-9c2b-6addfcbf987c.png">



## Steps Required to Run Car Detection GUI

1. Install all the requirements to run Flask based web server
   pip3 install -r requirements.txt


2. Start the Flask web server by running
   python3 detect-flask.py

   Go to the URL : http://localhost:8090

3. ![image](https://user-images.githubusercontent.com/11522867/164389365-222fbafe-d95d-4bb7-882c-234238a98964.png)
    From the screenshot press Choose file and select any image from test folder
    For images selected from test folder its class will be displayed but for new images class will be shown as N/A
    Press Predict when image is selected
    
    Below is the result of predicted image . 
    Predicted output consists of 
    1. Image with bounding box
    2. Predicted class
    3. Confidence Score
    
    <img width="558" alt="image" src="https://user-images.githubusercontent.com/11522867/164390065-8958d162-83f6-43ce-befc-e7bb9c9afc87.png">
