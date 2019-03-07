# wikihow-data
Clean, process and format the WikiHow dataset (https://github.com/mahnazkoupaee/WikiHow-Dataset) to work with the pointer generator network (Pytorch implementation: https://github.com/atulkum/pointer_summarizer).

## Steps
1. Put the WikiHowSep.csv dataset within this repository directory
2. Run clean.py
3. Run process.py
4. Run split.py (optional, can use same pre-split titles in all_train.txt, all_test.txt and all_val.txt)
5. Run format.py