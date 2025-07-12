# Intel_Unnati_Project
This project is a network traffic analysis tool that classifies network packets as **benign** or **malicious** using machine learning.  
It was developed as part of the **Intel® Unnati Industrial Training Program**.

You can view the deployed app on: 
```
[https://networkanalyse.streamlit.app/](https://networkanalyse.streamlit.app/)
```

---

## Features

- Detects and classifies network packets in real time.
- Uses a trained **Random Forest** model on the NSL-KDD dataset.
- Supports single and bulk predictions via a user-friendly **Streamlit** interface.
- Clean, modular architecture with reproducible training pipeline.

---

## Run Locally

Follow these steps to run the app on your local machine:

1. Clone the github repository in your local machine
2. Run the following command on your terminal to download all the dependencies
  ```
  pip install -r requirements.txt
  ```
3. Run this command in the terminal to start the application
  ```
  streamlit run app.py
  ```
The Dataset used to train the model is in the dataset directory.
The project report can be found in the docs directory.