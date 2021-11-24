# WSD_biencoder_reimplementation
This is a reproduction code base of the paper [Moving Down the Long Tail of Word Sense Disambiguation
with Gloss Informed Bi-encoders (2020 ACL)](https://blvns.github.io/papers/acl2020.pdf). Here is the [origin code base](https://github.com/facebookresearch/wsd-biencoders) of this paper provided by the author.


## Dependencies
The environment is the same as event-KG.

Please download the [WSD Evaluation Framework](http://lcl.uniroma1.it/wsdeval/) which contains the universal train/dev/test dataset and a JAVA evaluation script. Alternatively, you could download the [checkpoints](https://drive.google.com/file/d/1NZX_eMHQfRHhJnoJwEx2GnbnYIQepIQj/edit) given by authors. 

Note that 
- the Scorer.java in the WSD Framework data files needs to be compiled, with the Scorer.class file in the original directory of the Scorer file.
- The model architecture is a little difference between the original authors' and ours. If you want to use that checkpoint directly, execute ```generate_ckpt.ipynb``` and generate a pytorch state_dict compatible with our model in the top dictionary.

```
WSD_biencoder_reimplementation
  ├── wsd-biencoder         # checkpoints downloaded
  │   ├── ......
  │   └── best_model.ckpt
  |
  ├── WSD_Evaluation_Framework       # evaluation framework
  |
  └── best_model.ckpt       # generate from original checkpoint
```

## Run
Training:
```
python main.py --output_dir ./outputs
```

Official Evaluation on **dev** dataset:
```
java -cp ./WSD_Evaluation_Framework/Evaluation_Datasets/ Scorer ./WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt ./outputs/pred.txt
```
Current result (still running):
```
P=      73.2%
R=      73.2%
F1=     73.2%
```

Or you can use the author's checkpoint:
```
python main.py --output_dir ./outputs --inference_only
```
You'll get exactly the same result as reported in their paper.