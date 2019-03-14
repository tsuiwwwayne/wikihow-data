import random

NUM_TEST_LIMIT = 50000
NUM_VAL_LIMIT = 50000

with open('titles.txt', 'rb') as file_in:
    titles = [(random.random(), line) for line in file_in]
titles.sort()

with open('all_train.txt', 'wb') as file_train, \
     open('all_test.txt', 'wb') as file_test, \
     open('all_val.txt', 'wb') as file_val:

    for i, (_, line) in enumerate(titles):
        if i < NUM_TEST_LIMIT:
            file_test.write(line)
        elif i < (NUM_TEST_LIMIT + NUM_VAL_LIMIT):
            file_val.write(line)
        else:
            file_train.write(line)
