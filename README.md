Project Overview:

This project involves the creation of an Artificial Intelligence bot to play the game of "Tron". This bot uses A/B pruning with a modified Voronoi heuristic. NOTE: The included opponents (i.e TA bot) can only be played against in Mac/Linux machines

Instructions to Run:
NOTE - If you are running a Linux machine, change the ta_bots_linux.so to ta_bots.so and delete the original TA bot file

gamerunner.py can be used to test the AI bot against a series of opponent bots. This function takes a few command line arguments, the most important of which are:
 -bots lets you specify which bots to play against one another. The syntax is -bots <bot1> <bot2>
 -map lets you select the map that the game is to be played on. The syntax is -map <path to map>
 -multi test lets you run the same game setup (choice of bots and map) multiple times. You may want to run multiple tests with the -no image flag, so the games are played more quickly. (Printing to the terminal slows things down.) To do so, use -multi test <number of games> -no image.
 -no color runs the game without coloring the board printout. You should use this option if coloring causes display issues.

For example, the StudentBot can be tested against RandBot on the joust map using:

```
python gamerunner.py -bots student random -map maps/joust.txt
```

StudentBot can be tested against against WallBot 100 times with no visualizer on the empty room map with
python gamerunner.py -bots student wall -map maps/empty room.txt -multi test 100 -no image
Note: When running multiple tests, the StudentBot bot will move first in every other match.

Opponents There are four sample opponent bots:
1. RandBot chooses uniformly at random among all actions that do not immediately lead to a loss.
2. WallBot hugs walls and barriers to use space efficiently.
3. TA-Bot1 and TA-Bot2 Two more sophisticated TA bots. These use A/B pruning variations

The code for RandBot and WallBot is in bots.py. The implementation of the other TA bots is not exposed. Instead, it is included as a compiled module, ta_bots.so. You can
still test your bot against these bots: when running gamerunner.py, use the -bots flag with ta1 or ta2 as an argument.

There are eight sample maps, available in the maps directory. Two are empty maps, one big (13x13) and one small (7x7). The names of these maps can be found by navigating to the directory itself