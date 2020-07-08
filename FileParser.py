import librosa
import os
import csv


class FileParser:

    """ Description: A class for extracting data from a directory containing multiple files of the same type
        and writing the data to another file for further processing and manipulation. Currently can only extract data
        from .wav files (audio files) and can only write the data to a .csv file(s).

        Dependencies: librosa, os, csv

        Compatible Audio File Types (by file extension): .wav

        Features Extracted: MFCC, Spectral Centroid and Spectral Flatness. See librosa documentation for further details
        on these audio features

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

    def __init__(self, dir, file_exten, sample_rate=44100):

        """
        Defines the location and type of the file that contains the data needs to be extracted

        Parameters
        ----------
            dir : str
                directory containing the files
            file_exten : str
                file extension of the files that are to be read. All files with this extension will be read.
            sample_rate : int, optional
                sample rate an audio file is to be sampled at. Defaults to 44100. It's assumed all audio files in the
                directory are at the same sample rate.

        """

        self.directory = dir
        self.file_extension = file_exten
        self.data = []
        self.sr = sample_rate
        for file in os.listdir(self.directory):
            if file.endswith(self.file_extension):
                temp, sr = librosa.core.load(os.path.join(self.directory, file), sr=sample_rate)
                self.data.append(temp)

    def mfcc_extract(self, n_fft, window_len, hop_len, destination_dir, filename, filetype=".csv"):

        """
        Computes 20 MFCC coefficients for each window in the data. Only works on audio files. Note that because each
        window has 20 coeffcients a new file of type filetype will be created for each audio file that is read.
        Assuming you are creating a .csv file the columns will correspond to the MFC Coefficients and the rows
        will correspond to the respective audio frame (window). Thus, the data for each audio file read will
        be stored in a unique file of type filetype where the naming scheme of these files is
        filename + filetype + i where i is an integer representing the order in which the files were read. Starting
        at 0 going to ... infinity? As of python 2.7 int is promoted to a long if this number becomes too big python
        evaluates to infinity and would result in undefined behavior

        MFCC are computed using librosa.feature.mfcc

        Parameters
        ----------
            n_fft : int
                number of samples to include in the fft. The greater the number of samples the better your low
                frequency resolution will be
            window_len : int
                number of sample in a window (frame)
            hop_len : int
                number of samples the window moves over for each computation
            destination_dir : str
                name of the directory the created files will be placed into. If directory does not exist it will be
                created
            filename : str
                name of the file(s). Recall that for MFCC data each audio file that is read will correspond to one
                audio file written. Columns in the file correspong to MFC coeffcients and rows correspond to each
                individual frame
            filetype : str, optional
                determines the type of file to be written by it's extension. Default is ".csv"
         """

        self.__checkpath(destination_dir)
        for i in range(0, len(self.data)):
            mfcc_data = librosa.feature.mfcc(self.data[i], sr=self.sr, n_fft=n_fft, win_length=window_len,
                                             hop_length=hop_len)
            if filetype == ".csv":
                self.__write_csv(mfcc_data.T, "mfcc", destination_dir, filename + str(i) + filetype)

    def centroid_extract(self, n_fft, window_len, hop_len, destination_dir, filename, filetype=".csv"):

        """
        Computes spectral centroid for each window of each audio file. Only works on audio files. One file is
        is generated. Columns are the spectral centroid for that window and the rows are the different audio files.

        Spectral Centroid is computed using librosa.feature.spectral_centroid

        Parameters
        ----------
            n_fft : int
                number of samples to include in the fft. The greater the number of samples the better your low
                frequency resolution will be
            window_len : int
                number of sample in a window (frame)
            hop_len : int
                number of samples the window moves over for each computation
            destination_dir : str
                name of the directory the created files will be placed into. If directory does not exist it will be
                created
            filename : str
                name of the file.
            filetype : str, optional
                determines the type of file to be written by it's extension. Default is ".csv"
                 """

        self.__checkpath(destination_dir)
        for i in range(0, len(self.data)):
            centroid_data = librosa.feature.spectral_centroid(self.data[i], sr=self.sr, n_fft=n_fft,
                                                              win_length=window_len, hop_length=hop_len)
            if filetype == ".csv":
                self.__write_csv(centroid_data, "centroid", destination_dir, filename + filetype)

    def spectral_flt_extract(self,n_fft, window_len, hop_len, destination_dir, filename, filetype=".csv"):

        """
        Computes spectral flatness for each window of each audio file. Only works on audio files. One file is
        is generated. Columns are the spectral flatness for that window and the rows are the different audio files.

        Spectral Centroid is computed using librosa.feature.spectral_flatness

        Parameters
        ----------
            n_fft : int
                number of samples to include in the fft. The greater the number of samples the better your low
                frequency resolution will be
            window_len : int
                number of sample in a window (frame)
            hop_len : int
                number of samples the window moves over for each computation
            destination_dir : str
                name of the directory the created files will be placed into. If directory does not exist it will be
                created
            filename : str
                name of the file.
            filetype : str, optional
                determines the type of file to be written by it's extension. Default is ".csv"
                 """

        self.__checkpath(destination_dir)
        for i in range(0, len(self.data)):
            flt_data = librosa.feature.spectral_flatness(self.data[i], n_fft=n_fft, win_length=window_len,
                                                         hop_length=hop_len)
            if filetype == ".csv":
                self.__write_csv(flt_data, "flt", destination_dir, filename + filetype)

    @staticmethod
    def __write_csv(data, data_type, destination_dir, filename):
        if data_type == "mfcc":
            with open(os.path.join(destination_dir,filename), "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                for coeff in range(0, len(data)):
                    writer.writerow(data[coeff])
        else:
            with open(os.path.join(destination_dir,filename), "a") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                for sample in range(0, len(data)):
                    writer.writerow(data[sample])

    @staticmethod
    def __checkpath(path):
        if os.path.exists(path):
            return
        else:
            os.mkdir(path)
            return










