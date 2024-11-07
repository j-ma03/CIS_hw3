import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import List, Any

"""
Code created by Edmund Sumpena and Jayden Ma
"""

class Dataloader():
    """
    Base Dataloader class that performs some basic functionalities:
        -  Read the data files as a .csv
        -  Store basic file properties given on the first line of the data files
        -  Retrieve the individual coordinate points as an array of tuples
    """
    def __init__(
        self,
        metadata: List[Any],
        raw_data: NDArray[np.float32]
    ) -> None:
        
        # Remove any leading or trailing spaces
        for i in range(len(metadata)):
            metadata[i] = metadata[i].strip()

        # Stores metadata of the data file
        self.metadata: List[Any] = metadata

        # Store tuples of (x, y, z) coordinates read from the file
        self.raw_data: NDArray[np.float32] = raw_data

    # Construct a Dataloader class given a data file
    @staticmethod
    def read_file(filename: str) -> 'Dataloader':
        # Read the file line by line
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Extract metadata from the first line
        metadata = lines[0]

        # Extract raw data from the remaining lines
        raw_data = lines[1:]
        # for line in lines[1:]:
        #     raw_data.append([float(x) for x in line.strip().split(',')])

        # Convert raw_data to a numpy array
        raw_data = np.array(raw_data, dtype=np.float32)

        return Dataloader(metadata, raw_data)

    # Reads the data file's metadata from dataframe
    @staticmethod
    def _read_metadata(df: pd.DataFrame) -> List[Any]:
        return list(df.columns.values)

    # Reads (x, y, z) coordinates from each row
    @staticmethod
    def _read_raw_data(df: pd.DataFrame) -> NDArray[np.float32]:
        raw_data = []

        # Extract all the coordinates from dataframe
        for _, row in df.iterrows():
            raw_data.append(row.values.flatten().tolist()[0:3])

        return np.array(raw_data)
    
class RigidBodyDataloader(Dataloader):
    """
    Rigid body dataloader class that, in addition to the functionality of
    the base Dataloader, does the following:
        - extracts a set of coordinates corresponding to the LED markers in body coordinates
        - extracts a coordinate corresponding to the tip in body coordinates
    """    
    def __init__(
    self,
    metadata: List[Any],
    raw_data: NDArray[np.float32]
    ) -> None:
        super().__init__(metadata, raw_data)

        # number of LED markers in body coordinates
        self.N_markers: int = int(metadata[0])

    @staticmethod
    def read_file(filename: str) -> 'RigidBodyDataloader':
        dl = Dataloader.read_file(filename)
        return RigidBodyDataloader(dl.metadata, dl.raw_data)

class BodySurfaceDataloader(Dataloader):
    def __init__(
    self,
    metadata: List[Any],
    raw_data: NDArray[np.float32]
    ) -> None:
        super().__init__(metadata, raw_data)

    @staticmethod
    def read_file(filename: str) -> 'RigidBodyDataloader':
        dl = Dataloader.read_file(filename)
        return BodySurfaceDataloader(dl.metadata, dl.raw_data)

class SampleReadingsDataloader(Dataloader):
    def __init__(
    self,
    metadata: List[Any],
    raw_data: NDArray[np.float32]
    ) -> None:
        super().__init__(metadata, raw_data)

    @staticmethod
    def read_file(filename: str) -> 'RigidBodyDataloader':
        dl = Dataloader.read_file(filename)
        return SampleReadingsDataloader(dl.metadata, dl.raw_data)