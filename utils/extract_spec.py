"""
    Copyright 2018.4. Byeong-Yong Jang
    byjang@cbnu.ac.kr
    This code is for extracting spectrogram from wav-file.


    Input
    -----
    wav-file : path of wave file (16000 kHz)
    out-file : path of output file (pickle format)


    Options
    -------
    spec_type (string) : type of spectrogram
        tfspec      - compute log-spectrogram using 'tensorflow'
        tfmfcc      - compute mfcc using 'tensorflow'
        scispec     - compute log-spectrogram using 'scipy'
        rosaspec    - compute log-spectrogram using 'librosa'
        rosamelspec - compute mel-scale spectrogram using 'librosa'
        rosachroma  - compute chroma spectrogram using 'librosa'

    plot_spec (bool) : plot spectrogram using matplotlib


"""

from optparse import OptionParser

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pickle
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops

### hyper parameters ###
log_offset = 1e-6

### end of hyper parameter ###

# output spec data to pickle file
def output_data(filename,data):
    f = open(filename,'wb')
    pickle.dump(data,f)
    f.close()

# compute log-spectrogram using 'tensorflow' : 'tfspec'
def log_spec_tensorflow(wavfile,frame_size=400,frame_shift=160):
    sess = tf.InteractiveSession()
    wav_filename_placeholder = tf.placeholder(tf.string,[])

    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    wav_data = wav_decoder.audio

    spectrogram = contrib_audio.audio_spectrogram(
        wav_data,
        window_size=frame_size,
        stride=frame_shift,
        magnitude_squared=True)

    log_spectrogram = tf.log(spectrogram[0] + log_offset)

    log_spec_data = sess.run(log_spectrogram, feed_dict={wav_filename_placeholder: wavfile})

    return np.transpose(log_spec_data)


# compute mfcc using 'tensorflow' : tfmfcc
def mfcc_tensorflow(wavfile, frame_size=400, frame_shift=160 , order=13):
    sess = tf.InteractiveSession()
    wav_filename_placeholder = tf.placeholder(tf.string, [])

    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    wav_data = wav_decoder.audio

    spectrogram = contrib_audio.audio_spectrogram(
        wav_data,
        window_size=frame_size,
        stride=frame_shift,
        magnitude_squared=True)

    mfcc_ = contrib_audio.mfcc(spectrogram, wav_decoder.sample_rate, dct_coefficient_count=order)

    mfcc_data= sess.run(mfcc_, feed_dict={wav_filename_placeholder: wavfile})

    return mfcc_data


# compute log-spectrogram using 'scipy' : 'scispec'
def log_spec_scipy(wavfile,frame_size=400,frame_shift=160):
    sample_rate, data = scipy.io.wavfile.read(wavfile)
    # if nfft is 'None', fft size is 'nperseg'
    sample_freq, segment_time, spec_data = scipy.signal.spectrogram(data, fs=sample_rate,
                                                                    window='hann', nperseg=frame_size,
                                                                    noverlap=(frame_size - frame_shift), nfft=512,
                                                                    mode='psd')
    # mode = {psd, complex, magnitude, angle, phase}
    log_spec_data = np.log(spec_data + log_offset)
    return sample_freq, segment_time, log_spec_data


# compute log-spectrogram using 'librosa' : 'rosaspec'
def log_spec_librosa(wavfile,frame_size=400,frame_shift=160):
    data, fs = librosa.load(wavfile,sr=None)
    spec_data = librosa.core.stft(data,n_fft=512,hop_length=frame_shift,win_length=frame_size,
                                  window='hann',center=False)
    log_spec_data = np.log(np.abs(np.conj(spec_data)*spec_data*2) + log_offset)
    return np.transpose(log_spec_data)


# compute mel-scale spectrogram using 'librosa' : 'rosamelspec'
def mel_spec_librosa(wavfile,frame_size=400,frame_shift=160):
    data, fs = librosa.load(wavfile,sr=None)
    spec_data = librosa.core.stft(data,n_fft=512,hop_length=frame_shift,win_length=frame_size,
                                  window='hann',center=False)
    S = librosa.feature.melspectrogram(y=data,sr=fs,S=spec_data,
                                       n_mels=64,fmin=0.0,fmax=7600) # parameter for mel-filter
    log_S = np.log(S + log_offset)
    return np.transpose(log_S)

# compute chroma spectrogram using 'librosa' : 'rosachroma'
def chroma_spec_librosa(wavfile,frame_size=400,frame_shift=160):
    data, fs = librosa.load(wavfile, sr=None)
    spec_data = librosa.core.stft(data, n_fft=512, hop_length=frame_shift, win_length=frame_size,
                                  window='hann', center=False)
    chroma_data = librosa.feature.chroma_stft(sr=fs,S=spec_data,n_fft=512)
    return np.transpose(chroma_data)

def main():

    usage = "%prog [options] <wav-file> <out-file>"
    parser = OptionParser(usage)

    parser.add_option('--spec-type', dest='spec_type',
                      help='spectrogram type  [default: scispec ]',
                      default='scispec', type='string')
    parser.add_option('--plot-spec', dest='plot_spec',
                      help='plot spectrogram  [default: False ]',
                      default='False', type='string')

    (o, args) = parser.parse_args()
    (wav_path, out_file) = args

    spec_type = o.spec_type

    if spec_type == 'tfspec':
        spec_data = log_spec_tensorflow(wav_path)
    elif spec_type == 'tfmfcc':
        spec_data = mfcc_tensorflow(wav_path,order=13)
    elif spec_type == 'scispec':
        _, _, spec_data = log_spec_scipy(wav_path)
    elif spec_type == 'rosaspec':
        spec_data = log_spec_librosa(wav_path)
    elif spec_type == 'rosamelspec':
        spec_data = mel_spec_librosa(wav_path)
    elif spec_type == 'rosachroma':
        spec_data = chroma_spec_librosa(wav_path)
    else :
        _, _, spec_data = log_spec_scipy(wav_path)


    if o.plot_spec == 'True':
        plt.figure()
        plt.pcolormesh(spec_data)
        plt.title('Log spectrogram (direct) of tensorflow')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        plt.show()


    output_data(out_file,spec_data)

if __name__=="__main__":
    main()