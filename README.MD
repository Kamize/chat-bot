# Chat Bot

This is a simple chat bot application.

## Installation

Follow these instructions to set up the project on your local machine.

### Prerequisites

Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

### Steps

1.  **Clone the repository or download the source code.**

    If you have Git installed, you can clone the repository:
    ```bash
    git clone <repository-url>
    cd chat-bot
    ```
    *(Replace `<repository-url>` with the actual URL of the repository.)*

    Alternatively, you can download the source code as a ZIP file and extract it.

2.  **Create and activate a Miniconda environment.**

    Open your terminal or Anaconda Prompt and run the following commands:
    ```bash
    # Create a new environment in directory name 'chat-bot' with Python 3.10
    conda create --name chat-bot python=3.10 -y

    # Activate the newly created environment
    conda activate chat-bot
    ```

3.  **Install the required dependencies.**

    With the environment activated, install the necessary packages using pip:
    ```bash
    # Make sure you are in the project's root directory
    pip install -r requirements.txt
    ```
    *(Note: This assumes a `requirements.txt` file exists. If not, you'll need to install dependencies based on the project's imports.)*

4.  **Run the application.**

    Execute the main script to start the chat bot:
    ```bash
    streamlit run main.py
    ```
