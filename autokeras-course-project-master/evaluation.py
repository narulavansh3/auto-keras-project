import pickle

#from audio_classifier import AudioClassifier
#from audio_regressor import AudioRegressor
from text_classifier import TextClassifier
#from video_classifier import VideoClassifier


def evaluate(path, supervised):
    (x_train, y_train), (x_test, y_test) = pickle.load(open(path, 'rb'))

    supervised.fit(x_train, y_train, time_limit=5*60*60)
    supervised.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    return  supervised.evaluate(x_test, y_test)


if __name__ == '__main__':
#    print(evaluate('sample/audio_classification_data', AudioClassifier()))
#    print(evaluate('sample/audio_regression_data', AudioRegressor()))
#    print(evaluate('sample/video_classification_data', VideoClassifier()))
    print(evaluate('sample/text_classification_final.dms', TextClassifier())) # text_classification_raw_data_sample2 text_classification_final.dms
