"""Methods used to load Citi Bike simulation data from file."""

# Standard libs.
import os
import zipfile

# Third-party libs.
import numpy as np


# Filepaths for Citi Bike trip dataset statistics.
# Paths are relative to the project root directory.
TRIP_DATA_DIR = 'simcode/src/data/trip_statistics/'
TRIP_STATS_ZIP_FILENME = 'tripStatistics.zip'
TRIP_COUNT_FILENAME = 'tripCountData.npy'
TRIP_DURATION_FILENAME = 'Durations.npy'
DESTINATION_PROBS_FILENAME = 'destinationP.npy'


def loadTripStatistics(tripDataDir=TRIP_DATA_DIR):
    """Loads Citi Bike trip statistics."""

    # Check if files unzipped.
    if not os.path.exists(
            os.path.join(tripDataDir, TRIP_COUNT_FILENAME)):
        # Unzip files from archive.
        zipRef = zipfile.ZipFile(
            os.path.join(tripDataDir, TRIP_STATS_ZIP_FILENME), 'r')
        zipRef.extractall(tripDataDir)
        zipRef.close()
    # Load Citi bike trip statistics.
    tripCountData = np.load(
        os.path.join(tripDataDir, TRIP_COUNT_FILENAME))
    tripDurations = np.load(
        os.path.join(tripDataDir, TRIP_DURATION_FILENAME))
    destinationP = np.load(
        os.path.join(tripDataDir, DESTINATION_PROBS_FILENAME))

    return tripCountData, tripDurations, destinationP
