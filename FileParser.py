class FileParser:
    """ Description: A wrapper class for extracting data and writing the data to another file(s) for further processing
     and manipulation. Currently can extract data from .wav files (audio files) and the Spotify API. Can only write the
     data to .csv file(s).

        Dependencies: librosa, os, csv, spotipy,

        Compatible Audio File Types (by file extension): .wav

        As I expand this class should restructure into parent and child classes. I.e. parent class being something like
        FileParser and the children of FileParser being things like...AudioFileParser and XML File Parser

        ...

        Attributes
        ----------
        directory : str
            directory the files are located
        file_extension : str
            file extension of the files you would like read. Will read all files of this extension in the directory
            specified
        data : list
            list of data extracted from the file. In the case of an audio file it is the RAW floating point values of
            each sample
        sr : int
            defaults to 44100. Only used when reading and extracting data from audio files

        Methods
        -------
        mfcc_extract(n_fft, window_len, hop_len, destination_dir, filename, filetype):
            extracts MFCC data from the files and writes it to the file type determined by the extension provided in
            attribute filetype

        centroid_extract(n_fft, window_len, hop_len, destination_dir, filename, filetype):
            extracts spectral centroid data from the files and writes it to the file type determined by the extension
            provided in attribute filetype

        spectral_flt_extract(n_fft, window_len, hop_len, destination_dir, filename, filetype):
            extracts spectral flatness data from the files and writes it to the file type determined by the extension
            provided in attribute filetype
        """

    def __init__(self, path=None, file_exten=None, sample_rate=None):

        """
        Defines the location and type of the file that contains the data needs to be extracted

        Parameters
        ----------
            path : str
                directory containing the files
            file_exten : str
                file extension of the files that are to be read. All files with this extension will be read.
            sample_rate : int, optional
                sample rate an audio file is to be sampled at. Defaults to 44100. It's assumed all audio files in the
                directory are at the same sample rate.

        """

        import librosa.core.audio
        import os
        import csv

        self.librosa = librosa.core.audio
        self.os = os
        self.csv = csv
        self.directory = path
        self.file_extension = file_exten
        self.data = []
        if sample_rate is None and file_exten == ".wav":
            sample_rate = self.__get_sample_rate()

        self.sr = sample_rate
        if self.directory is not None and self.file_extension is not None:
            for file in self.os.listdir(self.directory):
                if file.endswith(self.file_extension):
                    temp, sr = self.librosa.load(self.os.path.join(self.directory, file))
                    self.data.append(temp)

    def extract_one_to_one(self, destination_dir, processing_function, param_dict, filename , filetype=".csv"):

        """
        Creates one output file for each input file. In the case of audio files this is designed for situations where
        you have more than one value for each window. MFCC for instance may have 20+ coeffcients for each frame in the
        input file. Assuming you are creating a .csv file the rows will correspond to the  individual frames and the
        respective values that correspond to a row will be stored in that rows columns. Thus, the data for each audio
        file read will be stored in a unique file of type filetype where the naming scheme of these files is
        filename + filetype + i where i is an integer representing the order in which the files were read. Starting
        at 0 going to ... infinity? As of python 2.7 int is promoted to a long if this number becomes too big python
        evaluates to infinity and would result in undefined behavior


        Parameters
        ----------
            destination_dir : str
                name of the directory the created files will be placed into. If directory does not exist it will be
                created
            processing_function : function
                function you wish to process the raw audio data with
            param_dict : dict
                dictionary of parameters that cor
            filename : str
                name of the file(s). Recall that for MFCC data each audio file that is read will correspond to one
                audio file written. Columns in the file correspong to MFC coeffcients and rows correspond to each
                individual frame
            filetype : str, optional
                determines the type of file to be written by it's extension. Default is ".csv"
         """

        self.__checkpath(destination_dir)
        for i, datum in enumerate(self.data):
            data = processing_function(datum, **param_dict)
            if filetype == ".csv":
                self.write_csv(data.T, "multi", destination_dir, filename + str(i) + filetype)

    def extract_all_to_one(self, destination_dir, processing_function, param_dict,filename, filetype=".csv"):

        """
        creates one output file that contains data from all the input files. In the case of audio files it is designed
        for situations where there is one number to represent each window.

        Parameters
        ----------
            destination_dir : str
                name of the directory the created files will be placed into. If directory does not exist it will be
                created
            processing_function : function
                function you wish to process the raw audio data with
            param_dict : dict
                dictionary of parameters that cor
            filename : str
                name of the file(s). Recall that for MFCC data each audio file that is read will correspond to one
                audio file written. Columns in the file correspong to MFC coeffcients and rows correspond to each
                individual frame
            filetype : str, optional
                determines the type of file to be written by it's extension. Default is ".csv"
                 """

        self.__checkpath(destination_dir)
        for datum in self.data:
            data = processing_function(datum, **param_dict)
            if filetype == ".csv":
                self.write_csv(data, "centroid", destination_dir, filename + filetype)

    def __get_sample_rate(self):
        file_list = self.os.listdir(self.directory)
        i = 0
        while True:
            if file_list[i].endswith(self.file_extension):
                sr = self.librosa.get_samplerate(self.os.path.join(self.directory, file_list[i]))
                break
            elif not file_list[i].endswith(self.file_extension):
                i += 1
            elif i > len(file_list):
                raise RuntimeError("Sample Rate was not specified and could not be determined")
        return sr

    def write_csv(self, data, data_type, destination_dir, filename, fieldnames=None):
        """
        writes data to .csv file. Either one or one csv file per file read

        Parameters
        ----------
            data : list, dict
                list or dictionary of the data to be writen. If dictionary data_type should be set to "dict" however the
                method will also do this check for you.
            data_type : str
                keyword string to inform the function of the type of data and how it should deal with it
            destination_dir : str
                absolute or relative path to where you would like the files to be written if the director does not exist
                it will be created for you
            filename : str
                name of the file(s). Recall that is data_type = "multi" data each  file that is read will correspond
                to one .csv file written and will be numbered starting at 0
            filetype : str, optional
                determines the type of file to be written by it's extension. Default is ".csv"
                 """
        if type(data) is dict and data_type != "dict":
            data_type = "dict"
        if type(data) is not dict and fieldnames is not None:
            raise RuntimeError("fieldnaames field only applies when using a dict data type for the input data")
        if data_type == "multi":
            with open(self.os.path.join(destination_dir,filename), "w") as csv_file:
                writer = self.csv.writer(csv_file, delimiter=",")
                for coeffs in data:
                    writer.writerow(coeffs)
        elif data_type == "dict":
            if fieldnames is None:
                fieldnames = list(data)
            with open(self.os.path.join(destination_dir,filename), "a") as csv_file:
                writer = self.csv.DictWriter(csv_file, fieldnames)
                writer.writeheader()
                for datum in data:
                    writer.writerow(datum)
        elif data_type != "multi" or data_type != "dict":
            raise RuntimeError("Supported data_types keywords are multi and dict")
        else:
            with open(self.os.path.join(destination_dir,filename), "a") as csv_file:
                writer = self.csv.writer(csv_file, delimiter=",")
                for frames in data:
                    writer.writerow(frames)

    def __checkpath(self, path):
        if self.os.path.exists(path):
            return
        else:
            self.os.mkdir(path)
            return


class SpotifyParsing():

    def __init__(self, client_id, client_secret):
        import spotipy as spot
        from spotipy.oauth2 import SpotifyClientCredentials
        import os
        import csv

        self.os = os
        self.csv = csv
        self.SpotifyClientCredentials = SpotifyClientCredentials
        self.spot = spot
        self.client_id = client_id
        self.client_secret = client_secret
        self.client_credentials_manager = self.SpotifyClientCredentials(client_id=self.client_id,
                                                                        client_secret=self.client_secret)
        self.sp = self.spot.Spotify(client_credentials_manager=self.client_credentials_manager)












