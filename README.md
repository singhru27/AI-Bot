## Project Name & Description

This project is a bot used to play the game of Tron for the Google Tron challenge. The rules of Tron are simple: Run into your trail or your opponent’s trail, or into a wall, and you die.

Support code (i.e a visualizer for the game as well as APIs for playing the game) have been provided by the organizers of the challenge. 

This project involves the creation of an Artificial Intelligence bot to play the game of "Tron". This bot uses A/B pruning with a modified Voronoi heuristic. NOTE: The included opponents (i.e Test1 bot) can only be played against in Mac/Linux machines


## Project Status

This project is completed

## Project Screen Shot(s)

![ScreenShot](https://github.com/singhru27/AI-Bot/blob/main/screenshots/Home.png?raw=true)


## Installation and Setup Instructions

To run the program, install the entire directory. Make sure you have python installed, and make sure to download all the dependencies in the 

```
requirements.txt
```
file. To test the bot, use the following command

```
python3 gamerunner.py -bots <bot1> <bot2>
```

The bots available to run are as follows:

```
student: The bot I have created
random: A bot that moves randomly
wall: A bot that uses a wall-hugging heuristic
test1: A bot that uses A/B pruning
test2: A bot that uses A/B pruning with a greater search depth
```

The default map is a 10 by 10 grid, but a series of maps are supported. To see the bots in action in a different map, use the following flag (maps are provided in the downloaded directory). There are eight sample maps, available in the maps directory. Two are empty maps, one big (13x13) and one small (7x7). The names of these maps can be found by navigating to the directory itself

```
 -map <path to map>
```

multi test lets you run the same game setup (choice of bots and map) multiple times. You may want to run multiple tests with the -no image flag, so the games are played more quickly. (Printing to the terminal slows things down.) To do so, use 

```
-multi test <number of games> -no image.
```

To run the game without coloring the board printout, use the following flag. You should use this option if coloring causes display issues.

```
no color 
```

For example, the StudentBot can be tested against RandBot on the joust map using:

```
python gamerunner.py -bots student random -map maps/joust.txt
```

StudentBot can be tested against against WallBot 100 times with no visualizer on the empty room map with

```
python gamerunner.py -bots student wall -map maps/empty room.txt -multi test 100 -no image
```

Note: When running multiple tests, the StudentBot bot will move first in every other match.

The code for RandBot and WallBot is in bots.py. The implementation of the other test bots is not exposed. Instead, it is included as a compiled module. 

