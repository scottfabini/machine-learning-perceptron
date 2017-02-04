#!/usr/bin/env python3

import numpy as np
import sys
import getopt


class Parser:
    def parse_file(self, input_filename, max_inputs_to_parse):
        file = open(input_filename, 'r')
        outputs = []
        inputs = []
        bias_input = 1.0
        for i, line in enumerate(file):
            if (i > max_inputs_to_parse):
                break  # for debug, limits number of images to process
            line = line.strip().split(',')
            line = [int(x) for x in line]
            outputs.append(line[0])
            inputs.append(np.array(line[1:]))
            inputs[i] = list([x/255 for x in inputs[i]])
            inputs[i].insert(0, bias_input)
        return np.array(inputs), np.array(outputs)

'''
    Incomplete. Return to later.

    def read_preparsed_files(self):
        images_file = open('images.csv', 'r')
        targets_file = open('targets.csv', 'r')
        images = []
        targets = []
        bias_input = 1.0
        for i, line in enumerate(images_file):
            line = line.strip().strip('[]').split(',')
            print(line)
            line = [int(x) for x in line]
            images.append(np.array(line[1:]))
            images[i] = list([x/255 for x in images[i]])
            images[i].insert(0, bias_input)
        for i, line in enumerate(targets_file):
            line = line.strip().split(',')
            line = [int(x) for x in line]
            targets.append(line[0])
        print(images)
        print(targets)
        return np.array(images), np.array(targets)
'''
if __name__ == "__main__":

    randomize = False
    count = 10000

    ifile = '/dev/null'
    ofile_images = '/dev/null'
    ofile_targets = '/dev/null'

    # process command line arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:h", ["o_images=", "o_targets=", "count="])
    except getopt.GetoptError:
        print('ParseFile.py -i <input_filename> -c <count_to_read> '
              '--o_images=<output_filename> --o_targets=<output_filename> ')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('ParseFile.py -i <input_filename> -c <count_to_read> '
                  '--o_images=<output_filename> --o_targets=<output_filename> ')
            sys.exit()
        elif opt == "-i":
            ifile = arg
        elif opt == "--o_images":
            ofile_images = arg
        elif opt == "--o_targets":
            ofile_targets = arg
        elif opt == "--count":
            randomize = True
            count = int(arg)

    print('Configuration: \n'
          'Input File: {0}\n' 'Output Images: {1}\n' 'Output Targets: {2}\n' 'Count: {3}'
          .format(ifile, ofile_images, ofile_targets, count))

    # parse input files
    parser = Parser()

    print('Begin Parsing Training Set...')
    training_images, training_targets = parser.parse_file(ifile, count)

    f = open(ofile_images, 'w+')
    f.write(str([value for value in training_images.tolist()]) + '\n')
    f.close()

    f = open(ofile_targets, 'w+')
    f.write(str([value for value in training_targets]) + '\n')
    f.close()

    print('Parsing File Complete.\n')
    sys.exit(0)
