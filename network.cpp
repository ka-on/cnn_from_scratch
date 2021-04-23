#include "network.h"

using namespace std;

const double C_ETA1 = 27 * 32;
const double C_ETA2 = 27 * 28;
const double X_ETA1 = 100;
const double X_ETA2 = 100;
const double GRADIENT_CONSTANT = 1;

Net::Net(vector<unsigned>& topology, double targetVal)
{
    this->initCL(); // Initialize convolutional layers
    this->initK();  // Initialize kernels


    unsigned short numberOfLayers = topology.size();

    // Create and fill the layers
    for (unsigned i = 0; i < topology[0]; i++){
        this->net_layers.InputLayer.push_back(InputNeuron(i, topology[1]));
    }

    for (unsigned i = 0; i < topology.size()-2; i++){
        this->net_layers.HiddenLayers.push_back(HiddenLayer());
        for (unsigned j = 0; j < topology[i+1]; j++) {
            Coordinate neuronCoordinates;
            neuronCoordinates.x = i;
            neuronCoordinates.y = j;
            this->net_layers.HiddenLayers.back().push_back(HiddenNeuron(neuronCoordinates, topology[i+2]));
        }
    }

    for (unsigned i = 0; i < topology.back(); i++){
        Coordinate neuronCoordinates;
        neuronCoordinates.x = topology.size()-2;
        neuronCoordinates.y = i;
        this->net_layers.OutputLayer.push_back(OutputNeuron(neuronCoordinates));
    }

    this->net_topology = topology;
    this->net_targetval = targetVal;
}


long double Net::getResult() {return this->net_result;}
double Net::getTargetval() {return this->net_targetval;}
void Net::setTargetval(double newTargetval) {this->net_targetval = newTargetval;}
LayerComposition& Net::getLayers() {return this->net_layers;}


short unsigned Net::feedForward()
{
    // EXCEPTION HANDLING
    /*if (this->net_topology[0] != inputVals.size()){
        //cerr << "Number of inputs does not match the number of input neurons" << endl;
        return 0;
    }*/

    // INPUT LAYER
    // Pass input values to the input layer and activate
    /*for (unsigned i = 0; i < inputVals.size(); i++) {
        this->net_layers.InputLayer[i].setOutput(inputVals[i]); // Pass
        //cout << "Input Neuron#" << i << " activated | Output: " << this->net_layers.InputLayer[i].getOutput() << endl;
    }*/

    // FEED FORWARD
    // Feed input layer to the first hidden layer
    cout << "[#";
    for (unsigned i = 0; i < this->net_topology[0]; i++) {
        vector<Connection> cnc = this->net_layers.InputLayer[i].gac();
        for (unsigned j = 0; j < cnc.size(); j++) {
            long double output = this->net_layers.InputLayer[i].getOutput();
            //cout << "Input Neuron#" << i << " accessing Hidden Neuron[" << cnc[j].coor.x << "][" << cnc[j].coor.y << "]" << " | Weight: " << cnc[j].weight << " | Output: " << output << " | Weighted Input: " << cnc[j].weight * output << endl;
            this->net_layers.HiddenLayers[cnc[j].coor.x][cnc[j].coor.y].pbi(cnc[j].weight * output);
        }
        this->net_layers.InputLayer[i].clearInputs(); // Clear
      }

    // Feed through the hidden layers
    for (unsigned i = 0; i < this->net_topology.size()-3; i++) {
        cout << "#";

        /*int normFac = 0;
        long double orgVal = (this->net_layers.HiddenLayers[i][0].ws() > 0) ? this->net_layers.HiddenLayers[i][0].ws() : this->net_layers.HiddenLayers[i][0].ws()*-1;

        while (orgVal > 1 || orgVal < 0.1) {
            if (orgVal > 1) {
                orgVal /= 10;
                normFac -= 1;
            } else if (orgVal < 0.1) {
                orgVal *= 10;
                normFac += 1;
            }
            cout << "ORGVAL: " << orgVal << endl;
        }*/

        for (unsigned j = 0; j < this->net_topology[i+1]; j++) {
            this->net_layers.HiddenLayers[i][j].ws();
            //orgVal = this->net_layers.HiddenLayers[i][j].ws();

            //this->net_layers.HiddenLayers[i][j].setInput(orgVal * pow(10, normFac));

            this->net_layers.HiddenLayers[i][j].activate(); // Activate
            //cout << "Hidden Neuron[" << i << "][" << j << "] activated | Input: " << this->net_layers.HiddenLayers[i][j].getInput() << " | Output: " << this->net_layers.HiddenLayers[i][j].getOutput() << endl;
            vector<Connection> cnc = this->net_layers.HiddenLayers[i][j].gac();
            for (unsigned k = 0; k < cnc.size(); k++) {
                long double output = this->net_layers.HiddenLayers[i][j].getOutput();
                //cout << "Hidden Neuron[" << i << "][" << j << "] accessing Hidden Neuron[" << cnc[k].coor.x << "][" << cnc[k].coor.y << "]" << " | Value: " << cnc[k].weight * output << endl;
                this->net_layers.HiddenLayers[cnc[k].coor.x][cnc[k].coor.y].pbi(cnc[k].weight * output); // Pass
            }
        this->net_layers.HiddenLayers[i][j].clearInputs(); // Clear
        }
    }

    // Feed last input layer to the output layer
    cout << "#";
    for (unsigned i = 0; i < this->net_topology[this->net_topology.size()-2]; i++) {
        this->net_layers.HiddenLayers.back()[i].ws();
        this->net_layers.HiddenLayers.back()[i].activate(); // Activate
        //cout << "Hidden Neuron[" << this->net_layers.HiddenLayers.size()-1 <<"][" << i <<"] activated | Input: " << this->net_layers.HiddenLayers.back()[i].getInput() << " | Output: " << this->net_layers.HiddenLayers.back()[i].getOutput() << endl;
        vector<Connection> cnc = this->net_layers.HiddenLayers.back()[i].gac();
        for (unsigned j = 0; j < cnc.size(); j++) {
            long double output = this->net_layers.HiddenLayers.back()[i].getOutput();
            //cout << "Hidden Neuron[" << this->net_layers.HiddenLayers.size()-1 <<"][" << i <<"] accessing Output Neuron#" << cnc[j].coor.y << " | Value: " << cnc[j].weight * output << endl;
            this->net_layers.OutputLayer[cnc[j].coor.y].pbi(cnc[j].weight * output); // Pass
        }
        this->net_layers.HiddenLayers.back()[i].clearInputs(); // Clear
    }

    // Activate the output layer (single neuron)
    cout << "#";
    for (unsigned i = 0; i < this->net_topology.back(); i++) {
        this->net_layers.OutputLayer[i].ws();
        this->net_layers.OutputLayer[i].activate(); // Activate
        //cout << "Output Neuron#" << i << " activated | Input#1: " << this->net_layers.OutputLayer[i].getInputs()[0] << " | Input#2: " << this->net_layers.OutputLayer[i].getInputs()[1] << " | Output: " << this->net_layers.OutputLayer[i].getOutput() << endl;
        //this->net_layers.OutputLayer[i].clearInputs(); // Clear
    }

    // GET RESULT
    this->net_result = this->net_layers.OutputLayer[0].getOutput();
    cout << "]" << endl;
    return 1;
}


short unsigned Net::backProp()
{
    // OUTPUT LAYER
    for (unsigned i = 0; i < this->net_layers.OutputLayer.size(); i++) {
        //cout << "Updating weights for Output Neuron#" << i << endl;
        this->net_layers.OutputLayer[i].updateWeights(this->net_layers.HiddenLayers.back(), this->net_targetval);
    }

    // Last hidden layer
    for (unsigned i = 0; i < this->net_layers.HiddenLayers.back().size(); i++) {
        long double one = this->net_layers.OutputLayer[0].getRE(); // Output neuron error
        long double input = this->net_layers.HiddenLayers.back()[i].getInput();
        //cout << "Updating weights for Hidden Neuron[" << this->net_layers.HiddenLayers.size()-1 << "][" << i << "]" << endl;
        long double peio = (input > 0) ? input : input*0.100; // Partial error of input to output
        long double delta = one * peio;
        this->net_layers.HiddenLayers.back()[i].setRE(delta);

        Coordinate neurCoor = this->net_layers.HiddenLayers.back()[i].getCoordinate();
        for (unsigned j = 0; j < this->net_layers.HiddenLayers[this->net_layers.HiddenLayers.size()-2].size(); j++) {
            long double partial_error = delta * this->net_layers.HiddenLayers[this->net_layers.HiddenLayers.size()-2][j].getOutput();
            long double weight_change = (eta/this->net_layers.HiddenLayers.back().size()) * partial_error;
            //cout << "    Weight updated for Hidden Neuron[" << this->net_layers.HiddenLayers.size()-2 << "][" << j << "] | Weight Change: " << weight_change << " | DELTA: " << delta << " | Old Weight: " << this->net_layers.HiddenLayers[this->net_layers.HiddenLayers.size()-2][j].gcbc(neurCoor.x, neurCoor.y).weight << endl;
            this->net_layers.HiddenLayers[this->net_layers.HiddenLayers.size()-2][j].gcbc(neurCoor.x, neurCoor.y).weight_change = weight_change;
        }
    }

    // HIDDEN LAYERS
    // Hidden layers except the first one
    for (unsigned i = this->net_layers.HiddenLayers.size()-2; i > 0; i--) {
        for (unsigned j = 0; j < this->net_layers.HiddenLayers[i].size(); j++) {
            //cout << "Updating weights for Hidden Neuron[" << i << "][" << j << "]" << endl;
            this->net_layers.HiddenLayers[i][j].updateWeights(this->net_layers.HiddenLayers[i-1], this->net_layers.HiddenLayers[i+1],  this->net_layers.HiddenLayers[i].size());
        }
    }

    // First hidden layer
    for (unsigned i = 0; i < this->net_layers.HiddenLayers[0].size(); i++) {
        long double input = this->net_layers.HiddenLayers[0][i].getInput();
        long double peio = pow(eulers, input) / pow(1 + pow(eulers, input), 2.0);
        long double delta = 0.0;
        for (unsigned j = 0; j < this->net_layers.HiddenLayers[1].size(); j++) {
            long double re = this->net_layers.HiddenLayers[1][j].getRE();
            //cout << "a : " << this->net_layers.HiddenLayers[0][i].gcbc(1, j).weight << endl;
            delta += re * peio * this->net_layers.HiddenLayers[0][i].gwbc(1, j);
        }
        //delta /= this->net_layers.HiddenLayers[1].size();
        this->net_layers.HiddenLayers[0][i].setRE(delta);
        //cout << "Updating weights for Hidden Neuron[0][" << i << "]" << endl;
        Coordinate neurCoor = this->net_layers.HiddenLayers[0][i].getCoordinate();
        for (unsigned k = 0; k < this->net_layers.InputLayer.size(); k++) {
            long double partial_error = this->net_layers.InputLayer[k].getOutput() * delta;
            long double weight_change = (eta/this->net_layers.HiddenLayers[0].size()) * partial_error; // eta = 0.5
            //cout << "    Weight updated for Input Neuron#" << k << " | Weight Change: " << weight_change << " | Delta: " << delta << " | Neuron output: " << this->net_layers.InputLayer[k].getOutput() << endl;
            this->net_layers.InputLayer[k].gcbc(neurCoor.x, neurCoor.y).weight_change = weight_change;
        }
    }

    // Input layer
    for (unsigned i = 0; i < this->net_layers.InputLayer.size(); i++) {
        long double delta = 0.0;
        for (unsigned j = 0; j < this->net_layers.HiddenLayers[0].size(); j++) {
            long double re = this->net_layers.HiddenLayers[0][j].getRE();
            delta += re * this->net_layers.InputLayer[i].gwbc(0, j) / GRADIENT_CONSTANT;
        }

        this->net_layers.InputLayer[i].setRE(delta);
    }

    // Update weights
    for (unsigned i = 0; i < this->net_topology[0]; i++) {
        for (unsigned j = 0; j < this->net_layers.InputLayer[i].gac().size(); ++j) {
            this->net_layers.InputLayer[i].gac()[j].weight -= this->net_layers.InputLayer[i].gac()[j].weight_change;
        }
    }

    for (unsigned i = 0; i < this->net_topology.size()-2; i++) {
        for (unsigned j = 0; j < this->net_layers.HiddenLayers[i].size(); j++) {
            for (unsigned k = 0; k < this->net_layers.HiddenLayers[i][j].gac().size(); k++) {
                this->net_layers.HiddenLayers[i][j].gac()[k].weight -= this->net_layers.HiddenLayers[i][j].gac()[k].weight_change;
            }
        }
    }

    for (unsigned i = 0; i < this->net_topology.back(); i++) {
        for (unsigned j = 0; j < this->net_layers.OutputLayer[i].gac().size(); j++) {
            this->net_layers.OutputLayer[i].gac()[j].weight -= this->net_layers.OutputLayer[i].gac()[j].weight_change;
        }
    }

    // Clear the inputs container of output neurons
    for (unsigned i = 0; i < this->net_layers.OutputLayer.size(); i++) {
        this->net_layers.OutputLayer[i].clearInputs();
    }

    return 1;
}


void Net::initCL()
{
    this->net_c_layers.push_back(Channel(28, 20));
    this->net_c_layers.push_back(Channel(32, 20));
    this->net_c_layers.push_back(Channel(32, 10));
    this->net_c_layers.push_back(Channel(64, 10));
    this->net_c_layers.push_back(Channel(64, 5));
}

void Net::initK()
{
    for (unsigned i = 0; i < 2; i++) {
        vector<Kernel> newSet;
        this->net_c_kernels.push_back(newSet);
    }

    for (unsigned i = 0; i < 32; i++){
        this->net_c_kernels[0].push_back(Kernel(28, 32));
    }

    for (unsigned i = 0; i < 64; i++) {
        this->net_c_kernels[1].push_back(Kernel(32, 64));
    }

}


Kernel& Net::gkbi(unsigned setIndex, unsigned kernelindex){return this->net_c_kernels[setIndex][kernelindex];}


void Net::ftic(DataFile& inputFile) // fill the input channel
{
    for (unsigned particle = 0; particle < 28; particle++) {
        for (unsigned i = 0; i < 20; i++) {
            for (unsigned j = 0; j < 20; j++) {
                for (unsigned k = 0; k < 20; k++) {
                    this->net_c_layers[0].setVals(particle, i, j, k, inputFile.dat_contents_4D[particle][i][j][k]);
                }
            }
        }
    }
}


void Net::Conv3D_28to32()
{
    this->net_c_layers[0].azb(); // Add zero border
    cout << "[";
    for (unsigned layer32 = 0; layer32 < 32; layer32++) { // 32 layers of the second channel
        cout << "#";
        //cout << "layer: " << layer32 << endl;
        //-----------------------------------------------//
        for (unsigned i = 0; i < 20; i++) {
            for (unsigned j = 0; j < 20; j++) {
                for (unsigned k = 0; k < 20; k++) { // 20x20x20 = 8000 inputs on each layer
                    //cout << "i: " << i << " j: " << j << " k: " << k << endl;
                    // calculating entries for 32 layers, 8000 entries each, for the second channel (20x20x20x32)
                    //----------------------------------------------//
                    long double newVal = 0.0;
                    for (unsigned layer28 = 0; layer28 < 28; layer28++) {
                        for (unsigned l = i; l < i+3; l++) {
                            for (unsigned m = j; m < j+3; m++) {
                                for (unsigned n = k; n < k+3; n++) { // 3x3x3=27 weight entries of the kernel
                                    //cout << "l: " << l << " m: " << m << " n: " << n << endl;
                                    //------------------------------------------//
                                    //if (j == 19) {cout << this->net_c_layers[0].getVals()[layer28][1][21].size() << endl;}
                                    long double channelVal = this->net_c_layers[0].gcvbi(layer28, l, m, n);
                                    long double weight = this->net_c_kernels[0][layer32].gkwbi(layer28, i+2-l, j+2-m, k+2-n);
                                    newVal += (channelVal * weight) + this->net_c_kernels[0][layer32].gkbbi(layer28);
                                }
                            }
                        }
                    }
                    this->net_c_layers[1].setVals(layer32, i, j, k, newVal);
                }
            }
        }
    }
    this->net_c_layers[0].rzb();
}


void Net::Conv3D_32to64()
{
    this->net_c_layers[2].azb(); // Add zero border
    cout << "[";
    for (unsigned layer64 = 0; layer64 < 64; layer64++) { // 64 layers of the third channel
        cout << "#";
        /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
        for (unsigned i = 0; i < 10; i++) {
            for (unsigned j = 0; j < 10; j++) {
                for (unsigned k = 0; k < 10; k++) { // 10x10x10 = 1000 inputs on each layer
                    // Calculating entries for 64 layers, 1000 entries each, for the fourth channel (10x10x10x64)
                    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
                    long double newVal = 0.0;
                    for (unsigned layer32 = 0; layer32 < 32; layer32++) {
                        for (unsigned l = i; l < i+3; l++) {
                            for (unsigned m = j; m < j+3; m++) {
                                for (unsigned n = k; n < k+3; n++) { // 3x3x3=27 weight entries of the kernel
                                    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
                                    long double channelVal = this->net_c_layers[2].gcvbi(layer32, l, m, n);
                                    long double weight = this->net_c_kernels[1][layer64].gkwbi(layer32, i+2-l, j+2-m, k+2-n);
                                    newVal += (channelVal*weight) + this->net_c_kernels[1][layer64].gkbbi(layer32);
                                }
                            }
                        }
                    }
                    this->net_c_layers[3].setVals(layer64, i, j, k, newVal);
                }
            }
        }
    }
    this->net_c_layers[2].rzb();
}


long double Net::getMax(long double pool[8])
{
    long double max = pool[0];
    for (unsigned i = 0; i < 8; i++) {
        max = (max > pool[i]) ? max : pool[i];
    }
    return max;
}

void Net::MaxPool3D(Channel& ch1, Channel& ch2)
{
    // Exception handling
    if ( (ch1.getDim() / 2 != ch2.getDim()) && (ch1.getNL() != ch2.getNL()) )
    {
      cerr << "Channel dimensions don't match" << endl; return;
    }

    for (unsigned layer2 = 0; layer2 < ch2.getNL(); layer2++) { // For all layers of ch1. Ex. 32 for ch[1]
        /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
        for (unsigned i = 0; i < ch2.getDim(); i++) {
            for (unsigned j = 0; j < ch2.getDim(); j++) {
                for (unsigned k = 0; k < ch2.getDim(); k++) { // For all inputs on each layer Ex. 20x20x20 = 8000 x 32 for ch[1]
                    long double newVal;
                    long double pool[8];
                        /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
                    for (unsigned l = i*2; l < i*2+2; l++) {
                        for (unsigned m = j*2; m < j*2+2; m++) {
                            for (unsigned n = k*2; n < k*2+2; n++) {
                                pool[4*((i*2+1)-l) + 2*((j*2+1)-m) + ((k*2+1)-n)] = ch1.gcvbi(layer2, l, m, n);
                            }
                        }
                    }
                    newVal = getMax(pool);
                    ch2.setVals(layer2, i, j, k, newVal);
                }
            }
        }
    }
}


vector<Channel>& Net::getCL() {return this->net_c_layers;}
Channel& Net::gchbi(unsigned index) {
    return this->net_c_layers[index];
}

void Net::flatten()
{
    for (unsigned layer = 0; layer < 64; layer++) {
        //cout << "layer: " << layer << endl;
        for (unsigned i = 0; i < 5; i++) {
            for (unsigned j = 0; j < 5; j++) {
                for (unsigned k = 0; k < 5; k++) {
                    //cout << "i: " << i << " | j: " << j << " | k: " << k << endl;
                    this->net_layers.InputLayer[layer*125 + i*25 + j*5 +k].setOutput(this->net_c_layers[4].gcvbi(layer, i, j, k));
                }
            }
        }
    }
}

void Net::sendData()
{
  for (unsigned layer = 0; layer < 64; layer++) {
      //cout << "layer: " << layer << endl;
      for (unsigned i = 0; i < 5; i++) {
          for (unsigned j = 0; j < 5; j++) {
              for (unsigned k = 0; k < 5; k++) {
                  //cout << "i: " << i << " | j: " << j << " | k: " << k << endl;
                  long double delta = this->net_layers.InputLayer[layer*125 + i*25 + j*5 +i].getRE();
                  this->net_c_layers[4].setVals(layer, i, j, k, delta);
              }
          }
      }
  }
}

void Net::maxPool3D_bp_5to10()
{
    for (unsigned layer = 0; layer < 64; layer++) { // For every layer of the last channel (5x5x5x64)
        for (unsigned i = 0; i < 5; i++) {
            for (unsigned j = 0; j < 5; j++) {
                for (unsigned k = 0; k < 5; k++) { // For each entry (125 on each layer)
                    double leakyDerivative = this->net_c_layers[4].gcvbi(layer, i, j, k);
                    /*~~~~~~~~~~~~~~~~~~~~~~~~~~*/
                    for (unsigned l = i*2; l < 2; l++) {
                        for (unsigned m = j*2; m < 2; m++) {
                            for (unsigned n = k*2; n < 2; n++) {
                                long double channelVal = this->net_c_layers[3].gcvbi(layer, l, m, n);
                                long double setValue = channelVal * leakyDerivative;
                                this->net_c_layers[3].setVals(layer, l, m, n, setValue);
                            }
                        }
                    }
                }
            }
        }
    }
}


void Net::maxPool3D_bp_10to20()
{
    for (unsigned layer = 0; layer < 32; layer++) {
        for (unsigned i = 0; i < 10; i++) {
            for (unsigned j = 0; j < 10; j++) {
                for (unsigned k = 0; k < 10; k++) {
                    double leakyDerivative = this->net_c_layers[2].gcvbi(layer, i, j, k);
                    /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
                    for (unsigned l = i*2; l < 2; l++) {
                        for (unsigned m = j*2; m < 2; m++) {
                            for (unsigned n = 0; n < 2; n++) {
                                long double channelVal = this->net_c_layers[1].gcvbi(layer, l, m, n);
                                long double setValue = channelVal * leakyDerivative;
                                this->net_c_layers[1].setVals(layer, l, m, n, setValue);
                            }
                        }
                    }
                }
            }
        }
    }
}

void Net::Conv3D_32to64_bp()
{
    // Add zero border to the third channel (10x10x10x32)
    this->net_c_layers[2].azb(); // 12x12x12x32

    // Calculate delta weights
    cout << "Calculating Delta Weights: [";
    for (unsigned kernel = 0; kernel < 64; kernel++) {
        cout << "#";
        /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
        for (unsigned layer = 0; layer < 32; layer++) {
            /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
            for (unsigned i = 0; i < 3; i++) {
                for (unsigned j = 0; j < 3; j++) {
                    for (unsigned k = 0; k < 3; k++) {

                      long double deltaWeight = 0.0;
                      /*~~~~~ For a field of size m x n x l;(ours is 12 x 12 x 12) */
                        for (unsigned l = i; l < 10+i; l++) { // m - 2 + i (2 because kernel is 3x3x3)
                            for (unsigned m = j; m < 10+j; m++) { // n - 2 + j
                                for (unsigned n = k; n < 10+k; n++) { // l - 2 + k (var names not related)
                                    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
                                    deltaWeight += this->net_c_layers[2].gcvbi(layer, l, m, n) * this->net_c_layers[3].gcvbi(kernel, l-i, m-j, n-k) / C_ETA1;
                                }
                            }
                        }
                        //cout << "DELTA WEIGHT: " << deltaWeight << endl;
                        //cout << "Delta Weight for [" << layer << "][" << i << "][" << j << "][" << k << "]: " << deltaWeight << " | Old Weight: " << this->net_c_kernels[1][kernel].gkwbi(layer, i, j, k) << endl;
                        this->net_c_kernels[1][kernel].sdwbi(layer, i, j, k, deltaWeight);
                    }
                }
            }
        }
    }
    cout << "]" << endl;
    this->net_c_layers[2].rzb(); // back to 10x10x10x32


    // Calculate delta bias
    cout << "Calculating Delta Bias:    [";
    for (unsigned i = 0; i < 32; i++) {
        cout << "#";
        for (unsigned j = 0; j < 64; j++) {
            long double deltaBias = this->net_c_layers[3].sumVals(i) / C_ETA1;
            //cout << "Backpropagation 64 to 32 | Kernel#" << i << "Filter#" << j << " | Delta Bias: " << deltaBias << endl;
            this->net_c_kernels[1][j].incBias(i, deltaBias);
        }
    }
    cout << "]" << endl;

    // Calculate delta X
    this->net_c_layers[3].azb(); // 12x12x12x64
    this->net_c_layers[2].clear();
    cout << "Calculating Delta X:       [";
    for (unsigned kernel = 0; kernel < 64; kernel++) {
        cout << "#";
        vector<Filter> rotatedKernel = this->net_c_kernels[1][kernel].rotate180();
          for (unsigned layer = 0; layer < 32; layer++) {
              for (unsigned i = 0; i < 10; i++) {
                  for (unsigned j = 0; j < 10; j++) {
                      for (unsigned k = 0; k < 10; k++) {
                          long double newVal = 0.0;
                          /*~~~~~~~~~~~~~~~~~~~~~~~~*/
                          for (unsigned l = 0; l < 3; l++) {
                              for (unsigned m = 0; m < 3; m++) {
                                  for (unsigned n = 0; n < 3; n++) {
                                      long double value = this->net_c_layers[3].gcvbi(layer, i+l, j+m, k+n);
                                      long double weight = this->net_c_kernels[1][kernel].gkwbi(layer, l, m, n);
                                      newVal += (value * weight) / X_ETA1;
                                  }
                              }
                          }
                          this->net_c_layers[2].incVals(layer, i, j, k, newVal);
                      }
                  }
              }
          }
    }
    this->net_c_layers[3].rzb(); // 10x10x10x64
}

void Net::Conv3D_28to32_bp()
{
    // Add zero border to the second channel (10x10x10x32)
    this->net_c_layers[0].azb(); // 22x22x22x32

    // Calculate delta weights
    cout << "Calculating Delta Weights: [";
    for (unsigned kernel = 0; kernel < 32; kernel++) {
        cout << "#";
        /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
        for (unsigned layer = 0; layer < 28; layer++) {
            /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
            for (unsigned i = 0; i < 3; i++) {
                for (unsigned j = 0; j < 3; j++) {
                    for (unsigned k = 0; k < 3; k++) {

                      long double deltaWeight = 0.0;
                      /*~~~~~ For a field of size m x n x l;(ours is 12 x 12 x 12) */
                        for (unsigned l = i; l < 20+i; l++) { // m - 2 + i (2 because kernel is 3x3x3)
                            for (unsigned m = j; m < 20+j; m++) { // n - 2 + j
                                for (unsigned n = k; n < 20+k; n++) { // l - 2 + k (var names not related)
                                    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
                                    deltaWeight += this->net_c_layers[0].gcvbi(layer, l, m, n) * this->net_c_layers[1].gcvbi(kernel, l-i, m-j, n-k) / C_ETA2;
                                    //cout << this->net_c_layers[0].gcvbi(layer, l, m, n) << " * " << this->net_c_layers[1].gcvbi(kernel, l-i, m-j, n-k) << endl;
                                }
                            }
                        }
                        //cout << "Delta Weight for [" << layer << "][" << i << "][" << j << "][" << k << "]: " << deltaWeight << " | Old Weight: " << this->net_c_kernels[0][kernel].gkwbi(layer, i, j, k) << endl;
                        this->net_c_kernels[0][kernel].sdwbi(layer, i, j, k, deltaWeight);
                    }
                }
            }
        }
    }
    cout << "]" << endl;
    this->net_c_layers[0].rzb(); // back to 10x10x10x32

    // Calculate delta bias
    cout << "Calculating delta bias :[" << endl;
    for (unsigned i = 0; i < 28; i++) {
        cout << "#";
        for (unsigned j = 0; j < 32; j++) {
            long double deltaBias = this->net_c_layers[3].sumVals(i) / C_ETA2;
            //cout << "Backpropagation 32 to 28 | Kernel#" << i << "Filter#" << j << " | Delta Bias: " << deltaBias << endl;
            this->net_c_kernels[1][j].incBias(i, deltaBias);
        }
    }

    // Calculate delta X
    /*this->net_c_layers[1].azb(); // 12x12x12x64
    this->net_c_layers[0].clear();
    cout << "Calculating Delta X:       [";
    for (unsigned kernel = 0; kernel < 32; kernel++) {
        cout << "#";
        vector<Filter> rotatedKernel = this->net_c_kernels[0][kernel].rotate180();
          for (unsigned layer = 0; layer < 28; layer++) {
              for (unsigned i = 0; i < 20; i++) {
                  for (unsigned j = 0; j < 20; j++) {
                      for (unsigned k = 0; k < 20; k++) {
                          long double newVal = 0.0;
                          //~~~~~~~~~~~~~~~~~~~~~~~~
                          for (unsigned l = 0; l < 3; l++) {
                              for (unsigned m = 0; m < 3; m++) {
                                  for (unsigned n = 0; n < 3; n++) {
                                      long double value = this->net_c_layers[1].gcvbi(layer, i+l, j+m, k+n);
                                      long double weight = this->net_c_kernels[0][kernel].gkwbi(layer, l, m, n);
                                      newVal += value * weight;
                                  }
                              }
                          }
                          this->net_c_layers[0].incVals(layer, i, j, k, newVal);
                      }
                  }
              }
          }
    }
    this->net_c_layers[1].rzb();  // 10x10x10x64*/
}


void Net::clear()
{
    for (unsigned i = 0; i < 5; i++) {
        this->net_c_layers[i].clear();
    }
}
