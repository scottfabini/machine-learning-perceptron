class Parser:
    def parse_file(self, input_filename):
        f = open(input_filename, 'r')
        outputs = []
        inputs = []
        bias_input = 1.0
        for i, line in enumerate(f):
            #if i > 100: break #for debug, limits number of images to process
            line = line.strip().split(',')
            line = [int(x) for x in line]
            outputs.append(line[0])
            inputs.append(line[1:])
            inputs[i] = [x/255 for x in inputs[i]]
            inputs[i].insert(0, bias_input)

        return inputs, outputs






