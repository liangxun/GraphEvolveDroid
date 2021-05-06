# GraphEvolveDroid
Android malware detection in the scenario of ecosystem evolution.

### run scripts

step1: construct knn-graph
```
python3 evoluNetwork.py
or 
python3 evolveNetwork.py
```

step2: train and test model
```
python3 tarin.py --view app_mamadroid_app --keyword mamadroid
```


The dataset used in this repo comes from [TESSERACT: eliminating experimental bias in malware classification across space and time.](https://dl.acm.org/doi/abs/10.5555/3361338.3361389). Make sure dataset is saved in correspondding dir configed by `setting.py` before runing scripts.