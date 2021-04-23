#include <iostream>
#include "network.cpp"
#include "network.h"
#include <time.h>
#include <string>

using namespace std;


int main()
{
    cout << "This version comes with 30 qgp and 30 not qgp files in each of the qgp and nqgp folders." << endl;
    cout << "Not all 10.000 data files were put into the folders because their collective size is around 4 GB," << endl;
    cout << "and would make the upload too big." << endl;
    cout << "There are 5000 qgp and 5000 nqgp files on OLAT:" << endl;
    cout << "Under \"Materialien\\Milestones\\Milestones\\qgp_dataset_full.zip\"." << endl;
	cout << "The zip folder containing 2500 of each is also included in this folder" << endl;
    cout << "If training with more files is needed, these data files can be put into the according qgp and nqgp files." << endl;
    cout << "Note that trying to read the 31. file where there are only 30 files in the folder will cause an error.\n" << endl;

    ////////////////////////////////////////////////////////////////////////////
    string input1;
    cout << "Enter 1 to train with QGP files.\nEnter 2 to train with NQGP files.\nEnter 3 to train with a random mix of the two.\nInput: ";
    cin >> input1;

    unsigned input2;
    cout << "Enter the number of files you would like to train with.\nEach file takes around a minute to process.\nInput:";
    cin >> input2;


    srand((double) time(NULL));

    vector<unsigned> topology{8000, 125, 75, 30, 15, 2, 1};

    Net myNet(topology, 0.0);

    for (unsigned i = 0; i < input2; i++) {
        unsigned counter_qgp = 0;
        unsigned counter_nqgp = 0;

        DataFile myFile(input1, i, counter_qgp, counter_nqgp);

        if (myFile.getIs_q()) {
            counter_qgp++;
        } else {
            counter_nqgp++;
        }

        myNet.setTargetval(myFile.getIs_q() ? 1.0 : 0.0);

        DataFile& ref = myFile;

        ////////////////////////////////////////////////////////////////////////

        cout << "> Filling the input channel\n" << endl;
        myNet.ftic(ref);

        /*cout << "Layer#0\n" << endl;
        myNet.gchbi(0).printChannelReduced();

        cout << "\nKernel#0\n" << endl; ///////////
        myNet.gkbi(0, 0).printKernel();*/ ///////////

        cout << "> Convolution 20x20x20x28 to 20x20x20x32" << endl;
        myNet.Conv3D_28to32();
        cout << "]\n" << endl;

        /*cout << "\nLayer#1\n" << endl; //////////
        myNet.gchbi(1).printChannelReduced();*/

        cout << "> Activating 20x20x20x32 with LReLU\n" << endl;
        myNet.gchbi(1).activate();

        /*cout << "Layer#1: Activated with LReLU\n" << endl; /////////
        myNet.gchbi(1).printChannelReduced();*/

        cout << "> Maxpooling 20x20x20x32 into 10x10x10x32\n" << endl;
        myNet.MaxPool3D(myNet.gchbi(1), myNet.gchbi(2));

        /*cout << "\nKernel#1\n" << endl; ///////
        myNet.gkbi(1, 0).printKernel(); ///////

        cout << "\nLayer#2\n" << endl; ///////
        myNet.gchbi(2).printChannel();*/

        cout << "> Convolution 10x10x10x32 to 10x10x10x64" << endl;
        myNet.Conv3D_32to64();
        cout << "]\n" << endl;

        /*cout << "\nLayer#3\n" << endl; /////////
        myNet.gchbi(3).printChannelReduced();*/

        cout << "> Activating 10x10x10x64 with LReLU\n" << endl;
        myNet.gchbi(3).activate();

        /*cout << "\nLayer#3: Activated with LReLu\n" << endl; ///////
        myNet.gchbi(3).printChannelReduced();*/

        cout << "> Maxpooling 10x10x10x64 into 5x5x5x64\n" << endl;
        myNet.MaxPool3D(myNet.gchbi(3), myNet.gchbi(4));

        /*cout << "\nLayer#4\n" << endl; ///////
        myNet.gchbi(4).printChannel(); */

        cout << "> Feeding the input layer\n" << endl;
        myNet.flatten();

        /*cout << "\nOutputs of input neurons in multi layer perception\n" << endl; ////
        for (unsigned i = 0; i < 8000; i++) { ////////
            cout << myNet.getLayers().InputLayer[i].getOutput() << "  ";////////
        } ////////
        cout << endl;*/


        cout << "> Feed forward\n" << endl;
        myNet.feedForward();

        cout << myNet.getLayers().OutputLayer[0].getInputs()[0] << " | " << myNet.getLayers().OutputLayer[0].getInputs()[1] << endl;

        cout << " >>>> NETWORK RESULT: " << myNet.getResult() << " | DESIRED RESULT: " << myNet.getTargetval() << endl;

        cout << "\n> Back propagation\n" << endl;
        myNet.backProp();

        /*if (i == 1) {
            for (unsigned j = 0; j < 8000; j++) {
                cout << myNet.getLayers().InputLayer[i].gac()[0].weight << endl;
            }
        }*/


        cout << "> Put delta values of the input neurons back into 5x5x5x64 channel\n" << endl;
        myNet.sendData();

        /*cout << "\nLayer#4\n" << endl;
        myNet.gchbi(4).printChannel();*/

        cout << "> Maxpool backpropagation 5x5x5x64 to 10x10x10x64\n" << endl;
        myNet.maxPool3D_bp_5to10();

        /*cout << "\nLayer#3\n" << endl;
        myNet.gchbi(3).printChannelReduced();*/

        cout << "> Convolution backpropagation 10x10x10x64 to 10x10x10x32" << endl;
        myNet.Conv3D_32to64_bp();
        cout << "]\n" << endl;

        /*cout << "Kernel#1" << endl;
        myNet.gkbi(1, 0).printKernel();

        cout << "\nLayer#2\n" << endl;
        myNet.gchbi(2).printChannel();*/

        cout << "> Maxpool backpropagation 10x10x10x32 to 20x20x20x32\n" << endl;
        myNet.maxPool3D_bp_10to20();

        /*cout << "\nLayer#1\n" << endl;
        myNet.gchbi(1).printChannelReduced();*/

        cout << "> Convolution backpropagation 20x20x20x32 to 20x20x20x28" << endl;
        myNet.Conv3D_28to32_bp();
        cout << "]\n" << endl;

        /*cout << "Kernel#0" << endl;
        myNet.gkbi(0, 0).printKernel();*/

        cout << "\nEnd\n\n\n" << endl;
    }
    return 0;
}
