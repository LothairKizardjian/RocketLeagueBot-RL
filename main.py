####

# to call this : python main.py --algorithm "algo name" test/train

import argparse

from src.bot import SimpleBot
from src import ConfigClass
from configparser import ConfigParser, ExtendedInterpolation

###########################
#####     PARSER      #####
###########################

parser = argparse.ArgumentParser(
    description="Implementation of the REINFORCE Algo to train Rocket League bot"
)

parser.add_argument(
    "-c", "--configfile", type=str, help="path of the config file", required=True
)

parser.add_argument(
    "-m", "--mode", type=str, help="mode, can be either train or test", default="train", required=True
)

parser.add_argument(
    "-r", "--retrain", type=int, help="1 to retrain a model from scratch, 0 to continue training on the model saved at save_path (see config file)", default=1
)

args = parser.parse_args()

config_file_path = args.configfile
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(config_file_path)
config = ConfigClass(config)

mode = args.mode
retrain = args.retrain

def main():
    bot = SimpleBot(config, mode, retrain)
    bot.run()
    
if __name__ == "__main__":
    main()