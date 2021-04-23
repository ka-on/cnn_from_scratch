#ifndef NEURON_H
#define NEURON_H

#include <vector>

using namespace std;
// Struct definitions

struct Coordinate
{
    /*
     * x is the layer of a neuron
     * y is the index of a neuron on its layer
     */

    unsigned short x;
    unsigned short y;
};


struct Connection
{
    Coordinate coor;
    long double weight;
    long double weight_change;
};

// Neuron declerations and layer composition
class InputNeuron;
class HiddenNeuron;
class OutputNeuron;

typedef vector<HiddenNeuron> HiddenLayer;

struct LayerComposition
{
    vector<InputNeuron> InputLayer;
    vector<HiddenLayer> HiddenLayers;
    vector<OutputNeuron> OutputLayer;
};

// Class definitions
class Neuron
{
    public:
        double getInput(); // Getter for input
        void setInput(long double newInput); // Setter for input
        vector<long double>& getInputs(); // Getter for inputs
        void pbi(long double newInput); // Push back inputs
        void clearInputs();

        long double getOutput(); // Getter for output
        void setOutput(long double newOutput); // Setter for output
        long double ws(); // Weighted sum

        Connection& gcbc(unsigned short x, unsigned short y); // Get connection by coordinate
        long double& gwbc(unsigned short x, unsigned short y); // Get weight by coordinate
        Connection& gcbi(unsigned short index); // Get connection by index
        long double& gwbi(unsigned short index); // Get weight by index

        vector<Connection>& gac(); // Get all connections
        void setWeight(Connection& connec, long double newWeight); // Setter for weight

        Coordinate getCoordinate(); // Getter for my coordinate

        virtual long double activate(); // Overwritten by inheriting classes

        long double cre(long double delta); // Calculate relative error
        long double getRE(); // Getter for relative error
        void setRE(long double newRE); // Setter for relative error
    protected:
        vector<long double> n_inputs; // Not weighted
        long double n_input; // Weighted (except for input neurons)
        long double n_output;
        long double n_relerr; // relative error
        vector<Connection> n_connections;
        Coordinate myCoor;
};


class InputNeuron: public Neuron
{
    public:
    InputNeuron(unsigned short myCoorY, unsigned sizeOfNextLayer);
    long double activate(); // Identity function, f(x) = x
};


class HiddenNeuron: public Neuron
{
    public:
    HiddenNeuron(Coordinate myCoor, unsigned sizeOfNextLayer);
    long double activate(); // Leaky RElU
    double leakyDer(long double arg); // Leaky RElU Derivative
    void updateWeights(HiddenLayer& previousLayer, HiddenLayer& nextLayer, unsigned sizeThisLayer);
};


class OutputNeuron: public Neuron
{
    public:
    OutputNeuron(Coordinate myCoor);
    long double activate(); // Softmax
    long double smd(); // Softmax partial derivative: e^(x+y) / (e^x + e^y)^2
    long double lossFunc(double targetVal); // 0.5*(target-output)^2
    void updateWeights(HiddenLayer& previousLayer, double targetVal);
};

#endif // NEURON_H
