#!/usr/bin/env python


import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops


wav_path = './sample_data/command_wav/on.wav'
sess = tf.InteractiveSession()
wav_filename_placeholder = tf.placeholder(tf.string,[])
wav_loader = io_ops.read_file(wav_filename_placeholder)
wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
wav_data = sess.run(wav_decoder,feed_dict={wav_filename_placeholder: wav_path}).audio

spectrogram = contrib_audio.audio_spectrogram(
        wav_data,
        window_size=400,
        stride=160,
        magnitude_squared=True)

mfcc_ = contrib_audio.mfcc(spectrogram,wav_decoder.sample_rate,dct_coefficient_count=13)

# extract spectrogram
spec_data = sess.run(spectrogram,feed_dict={wav_filename_placeholder: wav_path})

# extract mfcc
mfcc_data = sess.run(mfcc_,feed_dict={wav_filename_placeholder: wav_path})


