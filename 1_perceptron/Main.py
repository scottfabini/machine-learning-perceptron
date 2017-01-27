from Perceptron import Perceptron
from ParseFile import Parser
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# parse input files
parser = Parser()
print('Begin Parsing Training Set...')
training_images, training_targets = parser.parse_file('mnist_train.csv')
image_size = len(training_images[0])
number_of_training_images = len(training_images)
print('Parsing Training Set Complete.\n')

print('Begin Parsing Test Set...')
test_images, test_targets = parser.parse_file('mnist_test.csv')
number_of_test_images = len(test_images)
print('Parsing Test Set Complete.\n')

# initialize output files as blank files
f = open('accuracies.csv', 'w+')
f.close()
f = open('confusion.txt', 'w+')
f.close

learning_rates = [0.1, 0.01, 0.001]
for learning_rate in learning_rates:
    # initialize
    perceptron = Perceptron(image_size, learning_rate)

    training_predictions = [0] * number_of_training_images
    training_prediction_vectors = [0] * number_of_training_images
    training_accuracies = []
    training_accuracy = 0

    test_predictions = [0] * number_of_test_images
    test_prediction_vectors = [0] * number_of_test_images
    test_accuracies = []

    # begin learning
    print('Begin Learning With Learning Rate: {0}'.format(learning_rate))
    for epoch in range(70):
        # get training predictions
        for i in range(number_of_training_images):
            training_predictions[i] = perceptron.get_prediction(training_images[i])
        previous_epoch_training_accuracy = training_accuracy
        training_accuracy = accuracy_score(training_targets, training_predictions) * 100
        training_accuracies.append(training_accuracy)

        # get test predictions
        for i in range(number_of_test_images):
            test_predictions[i] = perceptron.get_prediction(test_images[i])
        test_accuracy = accuracy_score(test_targets, test_predictions) * 100
        test_accuracies.append(test_accuracy)

        # display intermediate results
        print('Epoch {0}'.format(epoch))
        print('Training Accuracy: {0:0.2f}%'.format(training_accuracy))
        print('Test Accuracy: {0:0.2f}%'.format(test_accuracy))
        print()

        # train based on predictions vs. targets
        for i in range(number_of_training_images):
            if training_predictions[i] != training_targets[i]:
                perceptron.train(training_predictions[i], training_targets[i], training_images[i])

        # break out of epoch loop if training accuracy > 80% and delta is small
        if training_accuracy > 80 and abs(previous_epoch_training_accuracy - training_accuracy) <= 1.0:
            break

    # display final results
    print('Learning Rate: {0}'.format(learning_rate))
    print('Training Accuracies per Epoch and Final Confusion Matrix: ')
    print(str([value for value in training_accuracies]))
    print(str(confusion_matrix(training_targets, training_predictions)) + '\n')

    print('Test Accuracies per Epoch and Final Confusion Matrix: ')
    print(str([value for value in test_accuracies]))
    print(str(confusion_matrix(test_targets, test_predictions)) + '\n\n')

    # write accuracy results to files
    f = open('accuracies.csv'.format(learning_rate), 'a')
    f.write('Learning Rate: {0}\n'.format(learning_rate))
    f.write('Training Accuracies:\n')
    f.write(str([value for value in range(len(training_accuracies))]).strip('[]') + '\n')
    f.write(str([value for value in training_accuracies]).strip('[]') + '\n\n')
    f.write('Test Accuracies:\n')
    f.write(str([value for value in range(len(test_accuracies))]).strip('[]') + '\n')
    f.write(str([value for value in test_accuracies]).strip('[]') + '\n\n\n')
    f.close()

    # write confusion matrix results to files
    f = open('confusion.txt', 'a')
    f.write('Learning Rate: {0}\n'.format(learning_rate))
    f.write('Training Confusion Matrix: \n')
    f.write(str(confusion_matrix(training_targets, training_predictions)) + '\n\n')
    f.write('Test Confusion Matrix: \n')
    f.write(str(confusion_matrix(test_targets, test_predictions)) + '\n\n\n')
    f.close()

