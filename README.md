### Simulating Human-like Reasoning with Large Models for Math Word Problem Solving

This is a repo for paper "[Simulating Human-like Reasoning with Large Models for Math Word Problem Solving]".

### Requirements
* python==3.7
* torch==1.9.0
* other pakages


### Environment
* OS: CentOS Linux release 7.7.1908
* CPU: 15 vCPU Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz
* GPU: A40 48GB
* CUDA: 11.1


### Datasets
* Math23K
```shell
test.json: three examples of data for Math23K
```
* MAWPS
```shell
dev.txt: three examples of data for MAWPS
```

### Running

**Train the model :** 
```shell
python run_seq2tree.py
```




