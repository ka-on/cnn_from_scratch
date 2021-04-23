#ifndef CHANNEL_H
#define CHANNEL_H

#include <vector>
#include <omp.h>
#include "datafile.h"
#include "datafile.cpp"

using namespace std;

struct Value{
    long double Val;
    long double deltaVal;
};

typedef vector<vector<vector<Value>>> Cube;

typedef vector<Cube> Layers4D;

struct Filter{
    Cube weights;
    long double bias;
};

class Kernel
{
    public:
        Kernel(unsigned fourthDim, double numChannel);
        long double weightGen(double numChannel);

        vector<Filter>& getKW(); // Get Kernel Weights
        long double gkwbi(unsigned layer, unsigned i, unsigned j, unsigned k); // Get Kernel Weight by Index
        long double gkbbi(unsigned layer); // Get Kernel Bias by Index
        void setBias(unsigned layer, long double newVal);
        void incBias(unsigned layer, long double newVal);

        vector<Filter> rotate180();

        void sdwbi(unsigned layer, unsigned i, unsigned j, unsigned k, long double newDW); // Set Delta Weight by Index
        void updateWeights();

        void clear();
        void printKernel();
    private:
        vector<Filter> k_weights_4D;
        vector<unsigned> k_dim;
};

class Channel
{
    public:
        Channel(unsigned numLayers, unsigned dim);

        Layers4D& getVals();
        long double gcvbi(unsigned layer, unsigned i, unsigned j, unsigned k);
        void setVals(unsigned layer, unsigned i, unsigned j, unsigned k, double newVal);
        void incVals(unsigned layer, unsigned i, unsigned j, unsigned k, double newVal); // Increase
        void incDV(unsigned layer, unsigned i, unsigned j, unsigned k, double newVal); // Increase Delta Val
        long double sumVals(unsigned layer);
        // add(?)

        unsigned getDim();
        unsigned getNL(); // Get Number of Layers

        long double LReLU(long double argInput);
        double LReLUDer(long double argInput); // LReLU derivative
        void activate(); // with LReLU

        void azb(); // Add Zero Border
        void rzb(); // Remove Zero Border

        void clear();
        void printChannel();
        void printChannelReduced();
        void itn(); // is there NaN
    private:
        bool c_zero_border; // True if a zero border is present
        Layers4D c_values_4D;
        unsigned c_numLayers;
        unsigned c_dim;
};

#endif // CHANNEL_H
