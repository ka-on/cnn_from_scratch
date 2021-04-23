#include "neuron.h"
#include <stdlib.h>
#include <math.h>

using namespace std;

// Constants
const double eulers = 2.71828;
const double eta = 0.5; // Learning rate
const double WEIGHTFACTOR = 1;


// METHODS FOR CLASS NEURON
// Input methods
double Neuron::getInput(){return this->n_input;}
void Neuron::setInput(long double newInput){this->n_input = newInput;}
void Neuron::pbi(long double newInput){this->n_inputs.push_back(newInput);}
void Neuron::clearInputs(){this->n_inputs.clear();}
vector<long double>& Neuron::getInputs(){return this->n_inputs;}

// Output methods
long double Neuron::getOutput(){return this->n_output;}
void Neuron::setOutput(long double newOutput){this->n_output = newOutput;}
long double Neuron::ws() // Weighted sum
{
    long double weighted_sum = 0.0;
    for (size_t i = 0; i < this->n_inputs.size(); i++) {
        weighted_sum += this->n_inputs[i];
    }
    this->n_input = weighted_sum;
    return weighted_sum;
}

// Connection methods
Connection& Neuron::gcbc(unsigned short x, unsigned short y)
{
    for (unsigned i = 0; i < this->n_connections.size(); i++) {
        if (this->n_connections[i].coor.x == x && this->n_connections[i].coor.y == y) {return n_connections[i];}
    }
}
long double& Neuron::gwbc(unsigned short x, unsigned short y)
{
    for (unsigned i = 0; i < this->n_connections.size(); i++) {
        if (n_connections[i].coor.x == x && n_connections[i].coor.y == y) {return n_connections[i].weight;}
    }
}
Connection& Neuron::gcbi(unsigned short index) {return this->n_connections[index];}
long double& Neuron::gwbi(unsigned short index) {return this->n_connections[index].weight;}
vector<Connection>& Neuron::gac(){return this->n_connections;}
void Neuron::setWeight(Connection& connec, long double newWeight){connec.weight = newWeight;}

// Self methods
Coordinate Neuron::getCoordinate(){return this->myCoor;}

// Methods for forward propagation
long double Neuron::activate(){this->n_output = this->n_input; return this->n_input;}

// Methods for weight update
long double Neuron::getRE()
{
    return this->n_relerr;
}

void Neuron::setRE(long double newRE)
{
    this->n_relerr = newRE;
}
// METHODS FOR CLASS INPUT NEURON
// Constructor
InputNeuron::InputNeuron(unsigned short myCoorY, unsigned sizeOfNextLayer)
{
    this->myCoor.x = 0;
    this->myCoor.y = myCoorY;

    for (unsigned i = 0; i < sizeOfNextLayer; i++) {
        this->n_connections.push_back(Connection());
        this->n_connections.back().coor.x = 0;
        this->n_connections.back().coor.y = i;
        this->n_connections.back().weight = ((double) rand() / (double) RAND_MAX) / sizeOfNextLayer;
    }
}

// Activation function
long double InputNeuron::activate() {
    this->n_output = this->n_inputs[0]; // f(x) = x
    return this->n_input;
}

// METHODS FOR CLASS HIDDEN NEURON
// Constructor
HiddenNeuron::HiddenNeuron(Coordinate myCoor, unsigned sizeOfNextLayer)
{
    this->myCoor.x = myCoor.x;
    this->myCoor.y = myCoor.y;

    for (unsigned i = 0; i < sizeOfNextLayer; i++) {
        this->n_connections.push_back(Connection());
        this->n_connections.back().coor.x = myCoor.x + 1;
        this->n_connections.back().coor.y = i;
        this->n_connections.back().weight = ((double) rand() / (double) RAND_MAX) / sizeOfNextLayer;
    }
}
// Activation function
long double HiddenNeuron::activate() {
    long double output = (this->n_input > 0) ? this->n_input : this->n_input * 0.100;
    this->n_output = output;
    if (output != output) {
         cout << "NaN found | Input was " << this->n_input << endl;
    }
    return output;
}

double HiddenNeuron::leakyDer(long double arg) {return (arg > 0) ? 1 : 0.100;}

// Update weights
void HiddenNeuron::updateWeights(HiddenLayer& previousLayer, HiddenLayer& nextLayer, unsigned sizeThisLayer) {

    long double din_dout = this->leakyDer(this->n_input); // #3
    long double delta = 0.0; // Increased later
    for (unsigned i = 0; i < nextLayer.size(); i++) {
        long double dout_din_output = this->gwbc(nextLayer[i].getCoordinate().x, nextLayer[i].getCoordinate().y); // #2
        long double delta_nextlayer = nextLayer[i].getRE(); // #1
        delta += din_dout * delta_nextlayer * dout_din_output;
    }
    //delta /= nextLayer.size();
    this->n_relerr = delta;
    for (unsigned i = 0; i < previousLayer.size(); i++) {
        long double dweight_din = previousLayer[i].getOutput();
        long double weight_change = (eta/sizeThisLayer) * dweight_din * delta;
        previousLayer[i].gcbc(this->myCoor.x, this->myCoor.y).weight = weight_change;
        //cout << "    Weight updated for Hidden Neuron[" << this->myCoor.x - 1 << "][" << i << "] | Weight Change: " << weight_change << " | Delta: " << delta << " | Neuron Output: " << previousLayer[i].getOutput() << endl;
    }
}

//-------------------------------------------------------------------------------
// METHODS FOR CLASS OUTPUT NEURON
// Constructor
OutputNeuron::OutputNeuron(Coordinate myCoor)
{
    this->myCoor.x = myCoor.x;
    this->myCoor.y = myCoor.y;
}

// Activation function
long double OutputNeuron::activate()
{
    int numOfOp = 0;
    long double magnitude1 = (this->n_inputs[0] > 0) ? this->n_inputs[0] : (-1 * this->n_inputs[0]);
    long double magnitude2 = (this->n_inputs[1] > 1) ? this->n_inputs[1] : (-1 * this->n_inputs[1]);
    long double bigger = (magnitude1 > magnitude2) ? magnitude1 : magnitude2;

    while (bigger > 10 || bigger < -0.01) {
        if (bigger > 10) {
            bigger /= 10;
            numOfOp -= 1;
        } else if (bigger < 0.5) {
            bigger *= 10;
            numOfOp += 1;
        }
    }

    this->n_inputs[0] *= pow(10, numOfOp);
    this->n_inputs[1] *= pow(10, numOfOp);

    long double output = pow(eulers, this->n_inputs[0]) / ( pow(eulers, this->n_inputs[0]) + pow(eulers, this->n_inputs[1]) );
    this->n_output = output;
    return output;
}

long double OutputNeuron::smd(){
    // Softmax derivative
    return pow(eulers, this->n_inputs[0] + this->n_inputs[1]) / pow((pow(eulers, this->n_inputs[0]) + pow(eulers, this->n_inputs[1])), 2.0);
}

long double OutputNeuron::lossFunc(double targetVal){
    // 0.5*(target-output)^2
    long double globalError = 0.5 * pow(targetVal - this->n_output, 2);
    return globalError;
}

// Update Weights
void OutputNeuron::updateWeights(HiddenLayer& previousLayer, double targetVal){
    long double dout_dloss = this->n_output - targetVal;
    long double din_dout = this->smd();

    long double delta = dout_dloss * din_dout;
    //cout << "SIZE: " << previousLayer.size() << endl;
    this->n_relerr = delta;
    for (unsigned i = 0; i < previousLayer.size(); i++) {
        long double dweight_din = this->n_inputs[i];
        //cout << "OUTPUT: " << previousLayer[i].getOutput() << endl;
        long double weight_change = eta * dweight_din * delta;
        previousLayer[i].gcbc(this->myCoor.x, this->myCoor.y).weight_change = weight_change;
        //cout << "    Weight updated for Hidden Neuron[" << this->myCoor.x - 1 << "][" << i << "] | Weight Change: " << weight_change << " | Delta: " << delta << " | Neuron Output: " << previousLayer[i].getOutput() << endl;
    }
}
