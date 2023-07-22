# Rochambeau - ML Game

![Rochambeau Logo](https://user-images.githubusercontent.com/88606859/255312709-2b886e88-ade9-4352-93d5-42ce5fe04755.png)
Rochambeau is a machine learning-powered game that allows users to play the classic game of "Stone, Paper, Scissors" with a twist. Instead of using traditional input methods, this game leverages the power of computer vision to let players use their hand gestures as the playing tool - Stone, Paper, or Scissors.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup](#setup)
- [How to Play](#how-to-play)
- [Play Demo Video](#play-demo-video)
## Introduction

Rochambeau is a fun and interactive machine learning game that showcases the potential of computer vision and artificial intelligence. The game is built using Django, a high-level web framework, and OpenCV, a popular computer vision library. With Rochambeau, players can experience an innovative twist to the traditional Stone, Paper, Scissors game by using their hand gestures captured through their camera.

## Features

- Play Stone, Paper, Scissors against the computer using hand gestures.
- Seamless integration of computer vision to recognize and interpret hand gestures.
- User-friendly interface for an engaging gaming experience.
- Real-time camera feed to capture and analyze hand gestures.
- Interactive computer opponent with varying levels of difficulty.
- Score tracking to keep track of your wins, losses, and ties.

## Technologies Used

- Django: A high-level Python web framework that simplifies web development.
- OpenCV: A popular computer vision library used for image and video processing.
- Python: The primary programming language used for developing the backend and machine learning algorithms.
- HTML/CSS/JavaScript: Used for creating the user interface and frontend elements.
- Bootstrap: A front-end framework for responsive and attractive UI design.

## Setup

To run Rochambeau on your local machine, follow these steps:

1. Clone the repository from GitHub:
```bash
git clone https://github.com/Naman82/SPS.git
cd SPS
```

2. Set up a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

4. Run the Django development server:

```bash
python manage.py runserver
```

5. Open your web browser and go to `http://localhost:8000` to access Rochambeau.

## How to Play

1. Launch Rochambeau->play->single player and allow camera access.
2. Click on S-P-S button and Position your hand in front of the camera, and the game will detect your gesture as one of the following: Stone, Paper, or Scissors.
3. The computer opponent will also make its choice simultaneously.
4. The game will determine the winner according to the classic rules: Stone beats Scissors, Scissors beats Paper, and Paper beats Stone.
5. Your score will be updated based on the game's outcome.
6. Play multiple rounds and try to beat the computer!


## Play Demo Video
[![Rochambeau Gameplay Video](https://user-images.githubusercontent.com/88606859/255312709-2b886e88-ade9-4352-93d5-42ce5fe04755.png)](https://youtu.be/2cmh1YycJHg)

We hope you enjoy playing Rochambeau as much as we enjoyed building it! If you have any questions or feedback, feel free to contact us at [naman@contact.com](mailto:b520031@iiit-bh.ac.in). Happy gaming!
