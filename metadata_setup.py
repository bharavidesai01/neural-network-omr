import os
import random

def write_list_to_file(items, filename):
    with open(filename, 'w') as f:
        for item in items:
            f.write("%s\n" % item)

def read_file_from_list(filename):
    with open(filename) as f:
        content = f.readline()
        return content.split('\t')

primus_directory = os.getcwd() + '/Data/Primus/'
sample_directories = [name for name in os.listdir(primus_directory) if os.path.isdir(os.path.join(primus_directory, name))]

# write the vocabulary set to a file
vocabulary = set()

for directory in sample_directories:
    vocabulary.update(read_file_from_list('{0}{1}/{2}.semantic'.format(
        primus_directory, 
        directory, 
        directory)))

vocabulary = list(vocabulary)
vocabulary.remove('')
vocabulary.sort()
write_list_to_file(vocabulary, 'Data/vocabulary_semantic.txt')


# generate training and testing 90/10 split
random.shuffle(sample_directories)
testing_samples = int(len(sample_directories) * .1)
write_list_to_file(sample_directories[testing_samples:], 'Data/train.txt')
write_list_to_file(sample_directories[:testing_samples], 'Data/test.txt')