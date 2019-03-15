# wikihow-data
Clean, process and format the WikiHow dataset (https://github.com/mahnazkoupaee/WikiHow-Dataset) to work with the pointer generator network (Pytorch implementation: https://github.com/atulkum/pointer_summarizer).

## Steps
1. Put the wikihowSep.csv dataset within this repository directory
2. Clean the WikiHow dataset.
```
python3 clean.py
```
3. Process the cleaned dataset.
```
python3 process.py
```
4. Split paragraph titles (optional, can use same pre-split titles in all_train.txt, all_test.txt and all_val.txt)
```
python3 split.py
```
5. Tokenise the paragraph files and write them in required input format for pointer summariser.
```
python3 format.py
```
