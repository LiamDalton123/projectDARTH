import logging
import math
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from AnalysisHandler import AnalysisHandler
from SoundDataSource import SoundDataSource
from SelectionHandler import SelectionHandler


def main():
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL.PngImagePlugin').disabled = True
    Tk().withdraw()
    filename = askopenfilename()
    sound_data_source = SoundDataSource(filename)
    selection_handler = SelectionHandler(sound_data_source)
    selection_handler.display_power_graph()
    last_selection = selection_handler.lastSelection
    logging.info("last selection was  " + str(last_selection))

    if last_selection is not None and last_selection[0] is not None and last_selection[1] is not None and \
            last_selection[1] > last_selection[0]:
        analysis_handler = AnalysisHandler(sound_data_source)
        analysis_handler.display_analysis(math.ceil(last_selection[0]), math.floor(last_selection[1]))
    else:
        logging.info("No valid selection made. Exiting.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
