import librosa
from FileParser import FileParser


test = FileParser("TestFiles/", ".wav", 10025)

test.extract_all_to_one("centroid_test3", librosa.feature.spectral_centroid,
                        {"sr": 10025, "win_length": 128, "hop_length": 16}, "samp")
test.extract_one_to_one("mfcc_test", librosa.feature.mfcc, {"sr": 10025, "win_length": 128, "hop_length":16 }, "samps")



