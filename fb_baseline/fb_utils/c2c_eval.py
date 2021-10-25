import torch
from BLEU import _bleu
import numpy as np

def eval_bleu(model, hidden_states, input_ids, beam_size, tokenizer, targets, max_length=170, path_gold='./test.gold', path_pred='./test.output'):
    preds = []
    m = torch.nn.LogSoftmax(dim=-1)
    p = []
    zero = torch.cuda.LongTensor(1).fill_(0)
    
    hidden_states = [torch.stack(x) for x in hidden_states]
    
    for i in range(input_ids.shape[0]):
        past_hidden = [x[:, i:i + 1].expand(-1, beam_size, -1, -1, -1) for x in hidden_states]
        beam = Beam(beam_size, tokenizer.bos_token_id, tokenizer.eos_token_id)
        for _ in range(max_length):
            if beam.done():
                break
            input_ids = beam.getCurrentState()
            logits, hidden = model('trans', input_ids=input_ids, eval_bleu=True, past=past_hidden)
            out = m(logits[:, -1, :]).data
            beam.advance(out)
            transform_out = [torch.stack(x) for x in hidden]
            past_hidden = [x.data.index_select(1, beam.getCurrentOrigin()) for x in transform_out]
        hyp = beam.getHyp(beam.getFinal())
        pred = beam.buildTargetTokens(hyp)[:beam_size]
        
        pred = [torch.cat([x.view(-1) for x in p] + [zero] * (max_length - len(p))).view(1, -1) for p in pred]
        p.append(torch.cat(pred, 0).unsqueeze(0))
    p = torch.cat(p, 0)
    for pred in p:
        t = pred[0].cpu().numpy()
        t = list(t)
        if 0 in t:
            t = t[:t.index(0)]
        text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
        preds.append(text)
        
    assert len(preds) == len(targets)
    EM = []
    with open(path_pred, 'w', encoding='utf-8') as f, \
            open(path_gold, 'w', encoding='utf-8') as f1:
        for pred, gold in zip(preds, targets):
            f.write(pred + '\n')
            f1.write(gold + '\n')
            EM.append(pred.split() == gold.split())
    bleu_score = round(_bleu(path_gold, path_pred), 2)
    EM = round(np.mean(EM) * 100, 2)
    return bleu_score, EM  
    

class Beam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence