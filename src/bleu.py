import sys
from nltk import bleu
import numpy as np

def read_ref(filename):
    f = open(filename,'r')
    ref_dacts = []
    ref_utts = []
    while True:
        dact = f.readline().strip()
        if not dact:
            break
        ref_dacts.append(dact)
        utts = []
        utt = f.readline().strip().split()
        while utt:
            utts.append(utt)
            utt = f.readline().strip().split()
        ref_utts.append(utts)
    f.close()
    return ref_dacts, ref_utts

def read_eval(filename):
    f = open(filename,'r')
    ref_dacts = []
    ref_utts = []
    det = False
    done = True
    while done:
        if not det:
            dact = f.readline().strip()
            dact = dact[1:]
        det = False
        if not dact:
            break
        ref_dacts.append(dact)
        utts = []
        utt = f.readline().strip().split()
        while True:
            utts.append(utt)
            utt = f.readline().strip().split()
            if not utt:
                utt = f.readline().strip()
                if not utt:
                    done = False
                    break
                if utt and utt[0] == '*':
                    det = True
                    dact = utt[1:]
                    break
                else:
                    utt = utt.split()
        ref_utts.append(utts)
    f.close()
    return ref_dacts, ref_utts

path = sys.argv[1]
version = sys.argv[2]
ref_dacts, ref_utts = read_ref('../ref_utterances/bleu_ref')
eval_dacts, eval_utts = read_eval(path+'bleu_eval/'+version)

ref_bleus = open(path+'bleu_eval_scores/'+version,'w')

bleu_scores = []
for i in range(len(eval_dacts)):
    dact = eval_dacts[i]
    ref_ind = ref_dacts.index(dact)
    ref_utt = ref_utts[ref_ind]

    for j in range(len(eval_utts[i])):
        eval_utt = eval_utts[i][j]
        score = bleu(ref_utt,eval_utt,[0.25,0.25,0.25,0.25])
        bleu_scores.append(score)
        #print(eval_utts[i])
        #print(eval_utt)
        ref_bleus.write(' '.join(eval_utt))
        ref_bleus.write(',')
        ref_bleus.write(str(score))
        ref_bleus.write('\n')
    ref_bleus.write('\n')
        #bleu_scores.append(bleu(eval_utt,ref_utt,[0.25,0.25,0.25,0.25]))
ref_bleus.close()

bleu_score = np.mean(bleu_scores)

f = open(path+'bleu/'+version,'w')
f.write(str(bleu_score))
f.close()
