#ifndef NETWORK_H
#define NETWORK_H

#include "neuron.h"
#include "neuron.cpp"
#include "channel.h"
#include "channel.cpp"


using namespace std;

typedef vector<Kernel> KernelSet;

// Class deinition network
class Net
{
    public:
        Net(vector<unsigned>& topology, double targetVal);

        long double getResult(); // Getter for result

        double getTargetval(); // Getter for the target value
        void setTargetval(double newTargetval); // Setter for the target value

        LayerComposition& getLayers();

        short unsigned feedForward();
        short unsigned backProp();

        void initCL(); // Initialize convolutional layer
        vector<Channel>& getCL(); // Get convolutional layer
        Channel& gchbi(unsigned index); // Get channel by index

        void initK();  // Initialize kernels
        Kernel& gkbi(unsigned setIndex, unsigned kernelindex); // Get kernel by index

        void ftic(DataFile& inputFile); // Fill the input channel

        void Conv3D_28to32();
        void Conv3D_32to64();
        long double getMax(long double pool[8]);
        void MaxPool3D(Channel& ch1, Channel& ch2);

        void flatten();
        void sendData();

        void maxPool3D_bp_5to10(); // Maxpool back propagation
        void maxPool3D_bp_10to20();

        void Conv3D_32to64_bp();
        void Conv3D_28to32_bp();

        void clear();
    protected:
        LayerComposition net_layers; // Normal(?) layers
        vector<Channel> net_c_layers; // Convolutional layers
        vector<KernelSet> net_c_kernels; // Kernels [0]:32, [1]:64

        vector<unsigned> net_topology;

        long double net_result;
        double net_targetval;
        long double net_error;
        long double net_delta;
        long double net_smdval;
};


#endif // NETWORK_H
