# 播放音频
import pygame.midi
import time
import json
import IO

with open("./GA.json",'r') as load_f:
    load_dict = json.load(load_f)
    num_tones = load_dict['num_tones']

pygame.midi.init()

# 与操作系统的音频合成器建立连接
player = pygame.midi.Output(pygame.midi.get_default_output_id())

# 播放单个音符
def play_one_note(num, dur):
    player.note_on(num, 64, 0)
    time.sleep(dur)
    player.note_off(num, 64, 0)

# 播放音乐序列
def play_sequence(seq):
    i = 0
    n = len(seq)
    while (i < n):
        if (seq[i] != 0):
            dur = 1
            x = seq[i]
            while (i < n-1 and seq[i+1] == num_tones-1):
                dur += 1
                i += 1
            play_one_note(x+52, 0.5*dur)
            #play_one_note(x+59, 0.5*dur)
        else:
            dur = 1
            while (i < n-1 and seq[i+1] == num_tones-1):
                dur += 1
                i += 1
            time.sleep(0.5*dur)
        i += 1

if (__name__ == '__main__'):
    play_sequence([8, 25, 25, 13, 8, 25, 6, 25, 5, 25, 3, 25, 1 ,25 ,25 ,25, 1 ,25 ,1 ,25, 3 ,25,5,25,5,25,1,25,5,25,6,25])
    #X, _ = IO.read_files('./data/chopin_nocturnes_train.txt')
    #print(X[0])
    #for x in X:
    #    print(x)
    #    play_sequence(x)