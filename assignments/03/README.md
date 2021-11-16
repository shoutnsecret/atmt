# ATMT Report 3: Improving Low-Resource NMT
Yifei Liu & Kexin Shi

The repository link is [https://github.com/kaikai23/atmt/](https://github.com/kaikai23/atmt/)

The report link is [https://github.com/kaikai23/atmt/edit/master/assignments/03/README.md](https://github.com/kaikai23/atmt/edit/master/assignments/03/README.md)

The task is to experiment with techniques for improving
low-resource NMT systems.

## Baseline

Instead of using the provided checkpoint as a baseline which produces a BLEU score of 16.8, we trained the model with default settings and used the generated best checkpoint to get a BLEU score of 15.1 (see atmt/assignments/03/lr=0.0003/lr=0.0003_result.png).

And we will compare other models with this 15.1 BLEU score.

## Code Changes

train command:
`python train.py --data data/en-fr/prepared/ --source-lang fr --target-lang en --save-dir assignments/03/lexical/checkpoints`
(before training first go to `lstm.py` and set `decoder_use_lexical_modle` to True)
