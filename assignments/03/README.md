# ATMT Report 3: Improving Low-Resource NMT
Yifei Liu & Kexin Shi

The repository link is [https://github.com/kaikai23/atmt/](https://github.com/kaikai23/atmt/)

The report link is [https://github.com/kaikai23/atmt/edit/master/assignments/03/README.md](https://github.com/kaikai23/atmt/edit/master/assignments/03/README.md)

The task is to experiment with techniques for improving
low-resource NMT systems.

## Baseline

Instead of using the provided checkpoint as a baseline which produces a BLEU score of 16.8, we trained the model with default settings and used the generated best checkpoint to get a BLEU score of 15.1 (see atmt/assignments/03/lr=0.0003/lr=0.0003_result.png).

And we will compare other models with this 15.1 BLEU score.

## Strategy

We choose to tune the hyperparameters (learning rate and batch size) and implement the lexical model, because we want to start with a simple setting and then do something more interesting.

## Data

We used the 10,000 parallel France-English data (/data/en-fr/). Since we choose to tune hyperparameters and implement the lexical model, we didn't make any changes to the preprocessing of the data. For training we directly used prepared data, and for testing we used raw data.

## Code Changes

For strategy one: tuning hyperparameters, we don't need to make any changes in the code. Just pass the arguments in the command line is enough.

For strategy two: implementing the lexical model, we made three changes in the `lstm.py` file.

1. initialize the one-hidden-layer FFNN and output linear layer

2. use the one-layer FFNN with skip connection to compute the h<sup>l</sup><sub>t</sub>

3. use the output linear layer to compute W<sup>l</sup>h<sup>l</sup><sub>t</sub> + b<sup>l</sup> and add it to the original decoder output.

For more detailed code changes please refer to the three TODO parts of `lstm.py`. And one last code change is setting the decoder_use_lexical_mode = True.

## Running experiments
We tried three different learning rates {0.001, 0.0003, 0.0001} and three different batch sizes {1, 2, 4}.

**lr=0.0001**:
train command:
`python train.py --data data/en-fr/prepared/ --source-lang fr --target-lang en --save-dir assignments/03/lr\=0.0001/checkpoints`

translate and evaluate command:
`translate.py --data data/en-fr/prepared/ --dicts data/en-fr/prepared/ --checkpoint-path assignments/03/lr=\0.0001/checkpoints/checkpoint_best.pt --output assignments/03/lr\=0.0001/translations.txt`,
`bash scripts/postprocess.sh assignments/03/lr=\0.0001/translations.txt assignments/03/lr\=0.0001/translations.p.txt en`,
`cat assignments/03/lr\=0.0001/translations.p.txt | sacrebleu data/en-fr/raw/test.en`

**lr=0.0003(default), lr=0.001, batch_size=1(default), batch_size=2, batch_size=4** are similar above.

**lexical model:**
First, go to `lstm.py` and set `decoder_use_lexical_modle` to True

train command:
`python train.py --data data/en-fr/prepared/ --source-lang fr --target-lang en --save-dir assignments/03/lexical/checkpoints`, 
other commands are similar.

## Results
We only select one hyperparameter each time and keep others fixed instead of searching all possible combinations.
For different learning rate:
|learning rates|batch size|lexical model|BLEU|1-gram|2-gram|3-gram|4-gram|
|:---|:---|:---|:---|:---|:---|:---|:---|
|0.0001|1|/|12.3|40.5|17.6|8.3|3.9|
|0.0003|1|/|15.1(baseline)|44.3|20.8|10.7|5.2|
|0.001|1|/|11.6|36.8|16.3|8.1|3.8|
|0.0003|2|/|19.4|49.6|25.4|14.3|7.9|
|0.0003|4|/|12.8|41.4|17.8|8.7|4.2|
|0.0003|1|Yes|19.5|49.6|26.0|14.0|8.0|
|0.0003|2|Yes|21.7|51.3|28.3|16.2|9.5|

We can see from the above table:

1. The default learning rate 0.0003 is the best among {0.0001, 0.0003, 0.001}
2. Batch size 2 is the best among {1, 2, 4}
3. Using lexical modle is better than not using it.

And our best model get a BLEU score of 21.7, which outperforms the baseline (15.1) very much.

What we learn from these experiments is that lexical model can help capture the meaning of sentences, and tuning hyperparameters can improve our model very much.
If we had time, next time we would like to also try the BPE method.
