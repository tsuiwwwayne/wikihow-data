'''
Reference: https://github.com/abisee/cnn-dailymail/blob/master/make_datafiles.py
The original code processes the data files for the CNN/DailyMail dataset.
This version works for the WikiHow dataset, specifically for formatting the
output files from process.py.
'''
import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
# Acceptable ways to end a sentence
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"',
              dm_single_close_quote, dm_double_close_quote, ")"]

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

all_train_titles = "all_train.txt"
all_test_titles = "all_test.txt"
all_val_titles = "all_val.txt"

paragraphs_dir = "paragraphs"
tokenized_paragraphs_dir = "paragraphs_tokenized"
finished_files_dir = "finished_files_paragraphs"
chunks_dir = os.path.join(finished_files_dir, "chunked")

# These are the number of paragraphs files we expect there to be in paragraphs
num_expected_paragraphs = 1263069

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # Number of examples per chunk, for the chunked data


def chunk_file(set_name):
    in_file = finished_files_dir + '/%s.bin' % set_name
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all():
    # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # Chunk the data
    for set_name in ['train', 'test', 'val']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name)
    print("Saved chunked data in %s" % chunks_dir)


def tokenize_paragraphs(paragraphs_dir, tokenized_paragraphs_dir):
    """Maps a whole directory of paragraph files to a tokenized version using Stanford CoreNLP Tokenizer"""
    print("Preparing to tokenize %s to %s..." % (paragraphs_dir, tokenized_paragraphs_dir))
    paragraphs = os.listdir(paragraphs_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for paragraph in paragraphs:
            f.write("%s \t %s\n" % (os.path.join(paragraphs_dir, paragraph), os.path.join(tokenized_paragraphs_dir, paragraph)))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing %i files in %s and saving in %s..." % (len(paragraphs), paragraphs_dir, tokenized_paragraphs_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")

    # Check that the tokenized paragraphs directory contains the same number of files as the original directory
    num_orig = len(os.listdir(paragraphs_dir))
    num_tokenized = len(os.listdir(tokenized_paragraphs_dir))
    if num_orig != num_tokenized:
        raise Exception("The tokenized paragraphs directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_paragraphs_dir, num_tokenized, paragraphs_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (paragraphs_dir, tokenized_paragraphs_dir))


def read_text_file(text_file):
    lines = []
    with open(text_file, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()


def get_title_hashes(title_list):
    return [hashhex(title.encode('utf-8')) for title in title_list]


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@summary" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    return line + " ."


def get_paragraph_summary(paragraph_file):
    lines = read_text_file(paragraph_file)

    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them
    lines = [fix_missing_period(line) for line in lines]

    # Separate out paragraph and abstract sentences
    paragraph_lines = []
    summary_lines = []
    is_next_summary = False
    for idx, line in enumerate(lines):
        if line == '':
          continue # empty line
        elif line.startswith("@summary"):
          is_next_summary = True
        elif is_next_summary:
          summary_lines.append(line)
        else:
          paragraph_lines.append(line)

    # Make paragraph into a single string
    paragraph = ' '.join(paragraph_lines)

    # Make summary into a single string, putting <s> and </s> tags around the sentences
    summary = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in summary_lines])

    return paragraph, summary


def write_to_bin(titles_file, out_file, makevocab=False):
    """Reads the tokenized paragraph files corresponding to the titles listed in titles_file and writes them to a out_file."""
    print("Making bin file for paragraphs listed in %s..." % titles_file)
    titles_list = read_text_file(titles_file)
    titles_hexdigests = get_title_hashes(titles_list)
    paragraph_file_names = [f + '.txt' for f in titles_hexdigests]
    num_paragraphs = len(paragraph_file_names)

    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        for idx, paragraph_file_name in enumerate(paragraph_file_names):
            if idx % 1000 == 0:
                print("Writing paragraph %i of %i; %.2f percent done" % (idx, num_paragraphs, float(idx)*100.0/float(num_paragraphs)))

            # Look in the tokenized story dirs to find the .story file corresponding to this url
            if os.path.isfile(os.path.join(tokenized_paragraphs_dir, paragraph_file_name)):
                paragraph_file = os.path.join(tokenized_paragraphs_dir, paragraph_file_name)
            else:
                print("Error: Couldn't find tokenized paragraph file %s in tokenized paragraphs directory %s. Was there an error during tokenization?" % (paragraph_file_name, tokenized_paragraphs_dir))
                # Check again if tokenized paragraphs directory contain correct number of files
                print("Checking that the tokenized paragraphs directory %s contain correct number of files..." % (tokenized_paragraphs_dir))
                check_num_paragraphs(tokenized_paragraphs_dir, num_expected_paragraphs)
                raise Exception("Tokenized paragraphs directory %s contain correct number of files but paragraph file %s was not found." % (tokenized_paragraphs_dir, paragraph_file_name))

            # Get the strings to write to .bin file
            paragraph, summary = get_paragraph_summary(paragraph_file)

            # Write to tf.Example
            tf_example = example_pb2.Example()
            # Key is 'article'/'abstract' because downstream pointer generator
            # code retrieves the paragraph/summary text via 'article'/'abstract' key
            tf_example.features.feature['article'].bytes_list.value.extend([paragraph.encode()])
            # key is 'abstract' because downstream pointer generator code
            tf_example.features.feature['abstract'].bytes_list.value.extend([summary.encode()])

            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # Write the vocab to file, if applicable
            if makevocab:
                paragraph_tokens = paragraph.split(' ')
                summary_tokens = summary.split(' ')
                summary_tokens = [t for t in summary_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
                tokens = paragraph_tokens + summary_tokens
                tokens = [t.strip() for t in tokens] # strip
                tokens = [t for t in tokens if t != ""] # remove empty
                vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


def check_num_paragraphs(paragraphs_dir, num_expected):
    num_paragraphs = len(os.listdir(paragraphs_dir))
    if num_paragraphs != num_expected:
        raise Exception("paragraphs directory %s contains %i files but should contain %i" % (paragraphs_dir, num_paragraphs, num_expected))


if __name__ == '__main__':
    # Check the paragraphs directory contain the correct number of paragraph files
    check_num_paragraphs(paragraphs_dir, num_expected_paragraphs)

    # Create some new directories
    if not os.path.exists(tokenized_paragraphs_dir): os.makedirs(tokenized_paragraphs_dir)
    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

    # Run stanford tokenizer on paragraphs directory, outputting to tokenized paragraphs directory
    tokenize_paragraphs(paragraphs_dir, tokenized_paragraphs_dir)

    # Read the tokenized paragraphs, do a little postprocessing then write to bin files
    write_to_bin(all_train_titles, os.path.join(finished_files_dir, 'train.bin'), makevocab=True)
    write_to_bin(all_test_titles, os.path.join(finished_files_dir, 'test.bin'))
    write_to_bin(all_val_titles, os.path.join(finished_files_dir, 'val.bin'))

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all()
