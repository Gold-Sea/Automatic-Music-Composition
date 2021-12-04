from mido import Message
from mido import MidiFile
from pymidifile import *
from reformat_midi import *
import numpy as np
def GetTrackName(track):
    """Extract the track name in a track.
    """
    for msg in track:
        if (msg.type == 'track_name'):
            return msg.name
    
    
def FromMIDIToChromosome(filename):
    """Convert a MIDI file into chromosome representation
    Parameters:
        filename: Path of the MIDI file. The file should be single-track and properly quantized.
    
    Return value:
        a list of numbers in the range 0..28, viz. the chromosome representation. 
        (Pitches beyond the range 1..27 are discarded.)
    [Note] Integer representation of pitches in Chromosome:
        0 = break,
        1 = F3, 2 = #F3, ... , 26 = #F5, 27 = G5
        28 = duration extension
        unit duration = an eighth note
    [Note] Integer representation of pitches as MIDI note number:
        60 = C4
        53 = F3, 54 = #F3, ... , 78 = #F5, 79 = G5
    [Note] Rule of convertion:
        Chromosome number + 52 == MIDI note number
    
    Notes
    """
    
    #mid = MidiFile(filename)
    #print(mid.ticks_per_beat)
    #for i,track in enumerate(mid.tracks):
    #    print(i,GetTrackName(track))
    #x = int(input())
    #for msg in mid.tracks[0]:
    #    print(msg)
    reformatted = reformat_midi(filename, verbose=False, write_to_file=False, override_time_info=True)
    matrix = mid_to_matrix(reformatted)
    quantizer = quantize_matrix(matrix, stepSize=0.5, quantizeOffsets=True, quantizeDurations=True)
    note_list = np.array(sorted(quantizer, key=lambda x: x[0], reverse=False))
    print(note_list[:,1]) # unit duration = quarter note
    print(note_list[:,2])
    print(note_list[:,1]+note_list[:,2])
    print(2*np.max(note_list[:,1]+note_list[:,2]))
    length = int(2*np.max(note_list[:,1]+note_list[:,2]))
    output = np.zeros(length)
    #for note in note_list:
    for note in sorted(quantizer, key=lambda x: x[0], reverse=False):
        #print(note)
        start = int(2*note[1])
        dur = int(2*note[2])
        pitch = int(note[0] - 52)
        if (pitch in range(1,28)):
            for t in range(start, start+dur):
                output[t] = pitch
    for i in range(length-1, 0, -1):
        if (output[i] == output[i-1]):
            output[i] = 28
    print(output.tolist())

#if __name__ == "__main__":
	
