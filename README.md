# Memory Lane - Helping Alzheimer’s patients remember

## Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/Spiral-Memory/MemoryLane.git
    ```

2. Navigate to the project directory:

    ```bash
    cd MemoryLane
    ```

3. Install the required dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Setting Up the Project

1. Navigate to the main file of the project and run the file.

## Usage

The main file of the project contains a Python script that interacts with an Assistance Bot. Here's how to use it:

1. Run the script:

    ```bash
    python main.py
    ```

2. You will see the Assistance Bot welcome message and a menu of options.

3. Input your choice by typing the corresponding number and pressing Enter:

    - To add a new relative, type `1`.
    - To start recognition, type `2`.
    - To exit the program, type `3`.

4. If you choose option `1` to add a new relative:
    - Enter the name, address, relationship, and gender of the relative when prompted.
    - Upload images of the relative as instructed.
    - The script will generate embeddings and update the face detector.

5. If you choose option `2` to start recognition:
    - Choose the mode by typing `1` for voice mode or `2` for text mode.
    - If you choose voice mode, the script will activate voice recognition.
    - If you choose text mode, the script will activate text recognition.

6. If you choose option `3`, the program will exit.

7. If you input an invalid choice, the program will display "Invalid Choice" and prompt you again.

Follow the on-screen instructions and prompts in the terminal to interact with the Assistance Bot.

