#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

class NeuralNetwork {
private:
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    float learningRate;

public:
    NeuralNetwork(float lr) : learningRate(lr) {
        srand(time(NULL));
        weights = {{(float)rand() / RAND_MAX, (float)rand() / RAND_MAX}};
        biases = {(float)rand() / RAND_MAX};
    }

    int activate(float sum) const {
        return sum > 0 ? 1 : 0;
    }

    int forwardPropagation(const std::vector<float>& input) const {
        if (input.size() != weights[0].size()) {
            std::cerr << "Input size does not match weight size!" << std::endl;
            return -1; // Error
        }

        float sum = biases[0];
        for (size_t i = 0; i < input.size(); ++i) {
            sum += input[i] * weights[0][i];
        }

        return activate(sum);
    }

    void train(const std::vector<std::vector<float>>& ANDdataset, const std::vector<int>& ANDlabels, int epochs) {
        if (ANDdataset.empty() || ANDlabels.empty() || ANDdataset.size() != ANDlabels.size()) {
            std::cerr << "Invalid training data!" << std::endl;
            return;
        }

        for (int epoch = 0; epoch < epochs; ++epoch) {
            float totalError = 0.0f;
            for (size_t i = 0; i < ANDdataset.size(); ++i) {
                float error = ANDlabels[i] - forwardPropagation(ANDdataset[i]);
                totalError += std::abs(error);

                for (size_t j = 0; j < weights[0].size(); ++j) {
                    weights[0][j] += learningRate * error * ANDdataset[i][j];
                }
                biases[0] += learningRate * error;
            }
            std::cout << "Epoch " << epoch + 1 << ", Total Error: " << totalError << std::endl;
        }
    }
};

int main() {
    std::vector<std::vector<float>> ANDdataset = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<int> ANDlabels = {0, 0, 0, 1};

    std::vector<std::vector<float>> ORdataset = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<int> ORlabels = {0, 1, 1, 1};

    std::vector<std::vector<float>> RandomDataset = {{0 , 1}, {1,0}, {0,0}, {1,0}, {1,1}, {0, 1}};
    std::vector<int> randomLables = {0, 1, 0, 0, 1, 0};

    NeuralNetwork nnAND(0.1f); 
    NeuralNetwork nnOR(0.1f); 
    NeuralNetwork nnRandom(0.1f); 

    int epochs = 1000;
    nnAND.train(ANDdataset, ANDlabels, epochs);
    nnOR.train(ORdataset, ORlabels, epochs);
    nnRandom.train(RandomDataset, randomLables, epochs);
    std::cout << "Results after training:" << std::endl;
    for (size_t i = 0; i < ANDdataset.size(); ++i) {
        std::cout << "Input: " << ANDdataset[i][0] << ", " << ANDdataset[i][1] << ", Output: " << nnAND.forwardPropagation(ANDdataset[i]) << std::endl;
    }
    std::cout << std::endl << std::endl;
    for(size_t i = 0; i < ORdataset.size(); ++i){
        std::cout << "Input: " << ORdataset[i][0] << ", " << ORdataset[i][1] << ", Output: " << nnOR.forwardPropagation(ORdataset[i]) << std::endl;

    }
    std::cout << std::endl << std::endl;
    for(size_t i = 0; i < RandomDataset.size(); ++i){
        std::cout << "Input: " << RandomDataset[i][0] << ", " << RandomDataset[i][1] << ", Output: " << nnRandom.forwardPropagation(RandomDataset[i]) << std::endl;
    }

    return 0;
}
