import scipy
import numpy as np
#from sklearn.metrics import confusion_matrix
import os
import sys
import warnings


def gender_voice_recognition(name):

    #Wczytanie pliku audio
    sampling_rate, signal = scipy.io.wavfile.read(name)

    #wybieramy sygnał z jednego kanału (może się zdarzyć że nagaranie posiadało dwa)
    if len(signal.shape) == 2:
        signal = [i[0] for i in signal]

    #Przemnozenie syganłu przez okno Hanninga
    signal = signal * np.hanning(len(signal))

    #Obliczenie spektrum sygnału
    signal_spectrum = abs(np.fft.fft(signal))

    freqs = [i / len(signal_spectrum) * sampling_rate for i in range(len(signal_spectrum))]

    #Usunięcie pierwszej dużej wartości (nwm jak to dokładnie nazwać, Sus to tłumaczył i nie pamiętam już)
    signal_spectrum[:np.where(np.array(freqs) >70)[0][0]] = 0

    #Obliczanie HPS (2,3 i 4 decymacja) i obliczenie częstotliwości bazowej
    fundamental_frequency = np.copy(signal_spectrum)
    for i in range(2, 5):
        hps = scipy.signal.decimate(signal_spectrum, i)
        fundamental_frequency[:len(hps)] *= hps

    #Znalezienie częstotliwości ostatecznej
    result = freqs[np.where(fundamental_frequency == max(fundamental_frequency))[0][0]]

    #Prosty test porównujący czestotliwość ostateczna z czestotliwoscia podstawowa tonu krtaniowego kobiety
    if result > 165:
        return 'K'
    else:
        return 'M'


def test_train_dataset():
    #Uruchomienie programu dla danych treningowych + macierz pomylek
    total = 0
    correct = 0
    true_values = []
    pred_values = []

    filenames = os.listdir('train')
    for filename in filenames:
        result = gender_voice_recognition('train/' + filename)
        print(result)
        if result == filename[4]:
            correct += 1
        total += 1
        true_values.append(filename[4])
        pred_values.append(result)

    print("Correct percent: ", correct / total)
    #print(confusion_matrix(true_values, pred_values, labels=["M", "K"]))


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    print(gender_voice_recognition(sys.argv[1]))
    #test_train_dataset()
