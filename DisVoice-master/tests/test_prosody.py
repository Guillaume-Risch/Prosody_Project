import os, sys
import pandas as pd
PATH=os.path.dirname(os.path.realpath(__file__))

PATH_DISVOICE=os.path.dirname(os.path.realpath(__file__))+"/disvoice/"
sys.path.append(PATH_DISVOICE)

import disvoice.prosody.prosody as prosody


def test_extract_prosody1():
    feature_extractor=prosody.Prosody()
    file_audio=PATH+"/../audios/sardoche_angry.wav"
    features1=feature_extractor.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
    print(features1.shape)

def test_extract_prosody2():
    feature_extractor=prosody.Prosody()
    path_audio=PATH+"/../audios/"
    features2=feature_extractor.extract_features_path(path_audio, static=True, plots=True, fmt="csv")
    print(features2)
    df = pd.DataFrame(features2)
    df.to_csv('happy.csv', index = False, encoding='utf-8')
    print(df)



def test_extract_prosody3():
    feature_extractor=prosody.Prosody()
    file_audio=PATH+"/../audios/OAF_bar_angry.wav"
    features3=feature_extractor.extract_features_file(file_audio, static=False, plots=False, fmt="torch")
    #print(features3.size())

if __name__ == "__main__":
    #test_extract_prosody1()
    test_extract_prosody2()
    #test_extract_prosody3()