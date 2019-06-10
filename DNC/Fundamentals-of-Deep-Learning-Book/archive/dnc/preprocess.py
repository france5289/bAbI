import sys
import pickle
import getopt
import operator
from urllib.request import urlopen
import tarfile
import numpy as np
from shutil import rmtree
import os
from os import listdir, mkdir
from os.path import join, isfile, isdir, dirname, basename, normpath, realpath, exists, getsize

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def create_dictionary(files_list):
    """
    creates a dictionary of unique lexicons in the dataset and their mapping to numbers

    Parameters:
    ----------
    files_list: list
        the list of files to scan through

    Returns: dict
        the constructed dictionary of lexicons
    """

    lexicons_dict = {}
    id_counter = 0

    llprint("Creating Dictionary ... 0/%d" % (len(files_list))) #40 ( train + test )

    for indx, filename in enumerate(files_list):
        #print("filename:", filename, "\n")
	
        with open(filename, 'r') as fobj:
            for line in fobj:
                #print(line, "\n")
                # first seperate . and ? away from words into seperate lexicons
                line = line.replace('.', ' .') # replace( old, new )
                line = line.replace('?', ' ?')
                line = line.replace(',', ' ')

                for word in line.split():
                    #print("word:", word)
					# give id to word
                    if not word.lower() in lexicons_dict and word.isalpha(): # isalpha mean all character
                        lexicons_dict[word.lower()] = id_counter
                        id_counter += 1

        llprint("\rCreating Dictionary ... %d/%d" % ((indx + 1), len(files_list)))

    print ("\rCreating Dictionary ... Done!")
    #print(lexicons_dict)
    return lexicons_dict


def encode_data(files_list, lexicons_dictionary, length_limit=None):
    """
    encodes the dataset into its numeric form given a constructed dictionary

    Parameters:
    ----------
    files_list: list
        the list of files to scan through
    lexicons_dictionary: dict
        the mappings of unique lexicons

    Returns: tuple (dict, int)
        the data in its numeric form, maximum story length
    """

    files = {}
    story_inputs = None
    story_outputs = None
    stories_lengths = []
    answers_flag = False  # a flag to specify when to put data into outputs list
    limit = length_limit if not length_limit is None else float("inf") # 正無窮
    #print("limit:", limit)
    llprint("Encoding Data ... 0/%d" % (len(files_list)))

    for indx, filename in enumerate(files_list):

        files[filename] = []

        with open(filename, 'r') as fobj:
            for line in fobj:

                # first seperate . and ? away from words into seperate lexicons
                line = line.replace('.', ' .')
                line = line.replace('?', ' ?')
                line = line.replace(',', ' ')

                answers_flag = False  # reset as answers end by end of line

                for i, word in enumerate(line.split()):

                    if word == '1' and i == 0:
                        # beginning of a new story
                        if not story_inputs is None:
                            #print(story_inputs) # a story convert the id
                            stories_lengths.append(len(story_inputs))
                            if len(story_inputs) <= limit:
                                files[filename].append({
                                    'inputs':story_inputs,
                                    'outputs': story_outputs
                                })
                        story_inputs = []
                        story_outputs = []

                    if word.isalpha() or word == '?' or word == '.':
                        if not answers_flag:
                            story_inputs.append(lexicons_dictionary[word.lower()])
                        else:
                            story_inputs.append(lexicons_dictionary['-']) # answer convert '-'
                            story_outputs.append(lexicons_dictionary[word.lower()]) 

                        # set the answers_flags if a question mark is encountered
                        if not answers_flag:
                            answers_flag = (word == '?')

        llprint("\rEncoding Data ... %d/%d" % (indx + 1, len(files_list)))

    print ("\rEncoding Data ... Done!")
    #print( "file:", files)
    return files, stories_lengths


if __name__ == '__main__':
    task_dir = dirname(realpath(__file__))
    #print(task_dir)
    options,_ = getopt.getopt(sys.argv[1:], '', ['length_limit='])
    #print(_)
    data_dir = join(task_dir, "/home/dchen/Desktop/bAbI/DNC/Fundamentals-of-Deep-Learning-Book/data/babi-en-10k")
    #print(data_dir)
    joint_train = True
    length_limit = None
    files_list = [] 

    if not exists(join(task_dir, 'data')):
        mkdir(join(task_dir, 'data')) # make a folder 

    for opt in options:
        if opt[0] == '--length_limit':
            length_limit = int(opt[1])
            #print("length :", length_limit)

    """if data_dir is None:
        raise ValueError("data_dir argument cannot be None")"""

    for entryname in listdir(data_dir):
        entry_path = join(data_dir, entryname)
        if isfile(entry_path):
            files_list.append(entry_path)

    
    #print("file:", files_list) # file path 
    #print("here:",len(files_list))
    lexicon_dictionary = create_dictionary(files_list)
    #print(lexicon_dictionary)
    lexicon_count = len(lexicon_dictionary)
    #print(lexicon_count) # 156 numbers
    # append used punctuation to dictionary
    lexicon_dictionary['?'] = lexicon_count
    lexicon_dictionary['.'] = lexicon_count + 1
    lexicon_dictionary['-'] = lexicon_count + 2

    # 159
    encoded_files, stories_lengths = encode_data(files_list, lexicon_dictionary, length_limit)

    #print(encoded_files)
    #print(stories_lengths)
    stories_lengths = np.array(stories_lengths)
    #print(stories_lengths) #convert matrix
    length_limit = np.max(stories_lengths) if length_limit is None else length_limit
    print ("Total Number of stories: %d" % (len(stories_lengths)))
    print ("Number of stories with lengthes > %d: %d (%% %.2f) [discarded]" % (length_limit, np.sum(stories_lengths > length_limit), \
             np.mean(stories_lengths > length_limit) * 100.0))
    print ("Number of Remaining Stories: %d" % (len(stories_lengths[stories_lengths <= length_limit]))) # remained story 

    processed_data_dir = join(task_dir, 'data', basename(normpath(data_dir)))
    train_data_dir = join(processed_data_dir, 'train')
    train_raw_data_dir = join(train_data_dir, 'train_raw_data') # nick
    test_data_dir = join(processed_data_dir, 'test')
    test_raw_data_dir = join(test_data_dir, 'test_raw_data') # nick
    if exists(processed_data_dir) and isdir(processed_data_dir):
        rmtree(processed_data_dir)

    mkdir(processed_data_dir)
    mkdir(train_data_dir)
    mkdir(test_data_dir)

    mkdir(train_raw_data_dir)
    mkdir(test_raw_data_dir)

    llprint("Saving processed data to disk ... ")
    # print(lexicon_dictionary)

    lexicon_sort = sorted(lexicon_dictionary.items(), key=lambda d: d[1])
    #print(lexicon_sort)
    # file write
    fp = open(join(processed_data_dir,"lexicon.txt"),"w", encoding='utf-8')
    for lexicon, number in lexicon_sort :
      fp.write( "lexicon:{}  number:{}\n".format( lexicon, number )) 
	
    pickle.dump(lexicon_dictionary, open(join(processed_data_dir, 'lexicon-dict.pkl'), 'wb'))
    
    joint_train_data = []

    for filename in encoded_files:
        if filename.endswith("test.txt"):
            
            encoded_files_dict = dict(enumerate(encoded_files[filename]))
            #print(encoded_files_dict)
            fp = open(join(test_raw_data_dir, basename(filename) + '.txt'), 'w', encoding='utf-8')
            for num,output in encoded_files_dict.items() :
              fp.write( "sentence:{}\n".format(output) )
            pickle.dump(encoded_files[filename], open(join(test_data_dir, basename(filename) + '.pkl'), 'wb'))
            
        elif filename.endswith("train.txt"):
            joint_train_data.extend(encoded_files[filename])
    #print(joint_train_data)
    pickle.dump(joint_train_data, open(join(train_data_dir, 'train.pkl'), 'wb'))
	
    fp = open(join(train_raw_data_dir,"train.txt"),"w", encoding='utf-8')
	
    joint_train_data_dict = dict(enumerate(joint_train_data))
    for input,output in joint_train_data_dict.items() :
       fp.write( "sentence:{}\n".format(output))

    llprint("Done!\n")
