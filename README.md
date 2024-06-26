# Medical Assistant Text Generator

This project is a Flask web application that uses a finetuned language model to generate possible diagnoses and treatment plans based on user-inputted symptoms. The application provides a web interface where users can enter symptoms either by typing or using voice input.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Features

- Web interface for inputting symptoms
- Voice input support using Speech Recognition
- Text generation using a finetuned language model
- Display generated diagnosis and treatment plans

## Installation


### Steps

1. Clone the repository:


2. Create and activate a virtual environment:


3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Ensure you have a GPU available for use of cuda

## Usage

1. Run the Flask application:
    ```sh
    python app.py
    ```

2. Open your web browser and navigate to:
    ```
    http://127.0.0.1:5000/
    ```

3. Enter symptoms in the provided text box or use the voice input feature, then click "Generate treatment plan".

## Project Structure

- `app.py`: The main Flask application file.
- `model.py`: Contains the function to generate text using the pretrained model.
- `templates/`: Directory containing HTML templates.
  - `index.html`: The home page where users input symptoms.
  - `result.html`: The page where the generated treatment plan is displayed.

