#include "channel.h"
#include <stdlib.h>
#include <math.h>

using namespace std;
const double WEIGHT_CONSTANT = 1;


long double Kernel::weightGen(double numChannel)
{
    double sign = (rand() % 2 == 1) ? -1 : 1;
    double signedPrefix = sign * ((double) rand() / (double) RAND_MAX);
    return (signedPrefix * (1 / sqrt(numChannel * 27)) ) * WEIGHT_CONSTANT;
}

              //           28                 32
Kernel::Kernel(unsigned fourthDim, double numChannel)
{
    this->k_dim.push_back(fourthDim);
    this->k_dim.push_back(numChannel);

    for (unsigned i = 0; i < fourthDim; i++){
        this->k_weights_4D.push_back(Filter());
        this->k_weights_4D.back().bias = this->weightGen(numChannel);
        for (unsigned j = 0; j< 3; j++){
            vector<vector<Value>> newVec2x;
            this->k_weights_4D.back().weights.push_back(newVec2x);
            for (unsigned k = 0; k < 3; k++){
                vector<Value> newVec1x;
                this->k_weights_4D.back().weights.back().push_back(newVec1x);
                for (unsigned l = 0; l < 3; l++){
                    this->k_weights_4D.back().weights.back().back().push_back(Value());
                    this->k_weights_4D.back().weights.back().back().back().Val = this->weightGen(numChannel);
                    this->k_weights_4D.back().weights.back().back().back().deltaVal = 0.0;
                }
            }
        }
    }

}

vector<Filter>& Kernel::getKW(){return this->k_weights_4D;}
long double Kernel::gkwbi(unsigned layer, unsigned i, unsigned j, unsigned k){
    return this->k_weights_4D[layer].weights[i][j][k].Val;
}

long double Kernel::gkbbi(unsigned layer){
    return this->k_weights_4D[layer].bias;
}

vector<Filter> Kernel::rotate180(){
    // save the original
    vector<Filter> rotatedCube = this->k_weights_4D;

    // rotate around y axis
    for (unsigned layer = 0; layer < this->k_dim[0]; layer++) {
        for (unsigned i = 0; i < 3; i++) {
            for (unsigned j = 0; j < 3; j++) {
                for (size_t k = 0; k < 3; k++) {
                    rotatedCube[layer].weights[i][j][k].Val = this->k_weights_4D[layer].weights[2-i][j][2-k].Val;
                }
            }
        }
    }

    // rotate around x axis
    for (unsigned layer = 0; layer < this->k_dim[0]; layer++) {
        for (unsigned i = 0; i < 3; i++) {
            for (unsigned j = 0; j < 3; j++) {
                for (size_t k = 0; k < 3; k++) {
                    rotatedCube[layer].weights[i][j][k].Val = this->k_weights_4D[layer].weights[i][2-j][2-k].Val;
                }
            }
        }
    }

    // rotate around z axis
    for (unsigned layer = 0; layer < this->k_dim[0]; layer++) {
        for (unsigned i = 0; i < 3; i++) {
            for (unsigned j = 0; j < 3; j++) {
                for (size_t k = 0; k < 3; k++) {
                    rotatedCube[layer].weights[i][j][k].Val = this->k_weights_4D[layer].weights[2-i][2-j][k].Val;
                }
            }
        }
    }

    return rotatedCube;
}

void Kernel::sdwbi(unsigned filter, unsigned i, unsigned j, unsigned k, long double newDW)
{
    this->k_weights_4D[filter].weights[i][j][k].Val = newDW;
}

void Kernel::clear()
{
    for (unsigned i = 0; i < this->k_dim[0]; i++) {
        for (unsigned j = 0; j < 3; j++) {
            for (unsigned k = 0; k < 3; k++) {
                for (unsigned l = 0; l < 3; l++) {
                    this->k_weights_4D[i].weights[j][k][l].Val = 0.0;
                }
            }
        }
    }
}

void Kernel::printKernel()
{
    for (unsigned i = 0; i < this->k_dim[0]; i++) {
        cout << "\n  Filter#" << i << " | Bias: " << this->k_weights_4D[i].bias << "\n" << endl;
        for (unsigned j = 0; j < 3; j++) {
            for (unsigned k = 0; k < 3; k++) {
                for (unsigned l = 0; l < 3; l++) {
                    cout << this->k_weights_4D[i].weights[j][k][l].Val << " ";
                }
                cout << endl;
            }
        }
    }
}

void Kernel::incBias(unsigned layer, long double newVal)
{
    this->k_weights_4D[layer].bias += newVal;
}

void Kernel::setBias(unsigned layer, long double newVal)
{
    this->k_weights_4D[layer].bias = newVal;
}
//------------------------------------------------------------------------------
Channel::Channel(unsigned numLayers, unsigned dim)
{
    this->c_numLayers = numLayers;
    this->c_dim = dim;
    for (unsigned i = 0; i < numLayers; i++){
        vector<vector<vector<Value>>> newVec3x;
        this->c_values_4D.push_back(newVec3x);
        for (unsigned j = 0; j < dim; j++){
            vector<vector<Value>> newVec2x;
            this->c_values_4D.back().push_back(newVec2x);
            for (unsigned k = 0; k < dim; k++){
                vector<Value> newVec1x;
                this->c_values_4D.back().back().push_back(newVec1x);
                for (unsigned l = 0; l < dim; l++){
                    this->c_values_4D.back().back().back().push_back(Value());
                    this->c_values_4D.back().back().back().back().Val = 0.0; // Initially
                    this->c_values_4D.back().back().back().back().deltaVal = 0.0; // Initially
                }
            }
        }
    }

    this->c_zero_border = false;
}

Layers4D& Channel::getVals(){return this->c_values_4D;}

long double Channel::gcvbi(unsigned layer, unsigned i, unsigned j, unsigned k) {
    return this->c_values_4D[layer][i][j][k].Val;
}

void Channel::setVals(unsigned layer, unsigned i, unsigned j, unsigned k, double newVal){
    this->c_values_4D[layer][i][j][k].Val = newVal;
}

void Channel::incVals(unsigned layer, unsigned i, unsigned j, unsigned k, double newVal){
    this->c_values_4D[layer][i][j][k].Val += newVal;
}

void Channel::incDV(unsigned layer, unsigned i, unsigned j, unsigned k, double newVal){
    this->c_values_4D[layer][i][j][k].deltaVal += newVal;
}

unsigned Channel::getDim(){return this->c_dim;}
unsigned Channel::getNL(){return this->c_numLayers;} // Get Number of Layers


long double Channel::LReLU(long double argInput)
{
    return (argInput > 0) ? argInput : argInput * 0.100;
}

double Channel::LReLUDer(long double argInput)
{
    return (argInput > 0) ? 1 : 0.100;
}

void Channel::activate()
{
    for (unsigned layer = 0; layer < this->c_numLayers; layer++) {
        for (unsigned i = 0; i < this->c_dim; i++) {
            for (unsigned j = 0; j < this->c_dim; j++) {
                for (unsigned k = 0; k < this->c_dim; k++) {
                    //cout << "before: " << this->c_values_4D[layer][i][j][k];
                    this->c_values_4D[layer][i][j][k].Val = this->LReLU(this->c_values_4D[layer][i][j][k].Val);
                    //cout << " | after: " <<  this->LReLU(this->c_values_4D[layer][i][j][k]) << endl;
                }
            }
        }
    }
}


void Channel::azb() // Add zero border
{
    // Exception handling
    if (this->c_zero_border) { cerr << "Zero border already present" << endl; return;}

    for (unsigned layer = 0; layer < this->c_numLayers; layer++) {
        for (unsigned i = 0; i < this->c_dim; i++) {
            for (unsigned j = 0; j < this->c_dim; j++) {
                this->c_values_4D[layer][i][j].push_back(Value());
                this->c_values_4D[layer][i][j].back().Val = 0.0;
                this->c_values_4D[layer][i][j].insert(this->c_values_4D[layer][i][j].begin(), Value());
                this->c_values_4D[layer][i][j][0].Val = 0.0;
            }
        }
    }

    for (unsigned layer = 0; layer < this->c_numLayers; layer++) {
        for (unsigned i = 0; i < this->c_dim; i++) {
            vector<Value> newVec1x;
            this->c_values_4D[layer][i].push_back(newVec1x);
            this->c_values_4D[layer][i].insert(this->c_values_4D[layer][i].begin(), newVec1x);
            for (unsigned j = 0; j < this->c_dim + 2; j++) {
                this->c_values_4D[layer][i].back().push_back(Value());
                this->c_values_4D[layer][i].back().back().Val = 0.0;
                this->c_values_4D[layer][i][0].push_back(Value());
                this->c_values_4D[layer][i][0].back().Val = 0.0;
            }
        }
    }

    for (unsigned layer = 0; layer < this->c_numLayers; layer++) {
        vector<vector<Value>> newVec2x;
        this->c_values_4D[layer].push_back(newVec2x);
        this->c_values_4D[layer].insert(this->c_values_4D[layer].begin(), newVec2x);
        for (unsigned i = 0; i < this->c_dim + 2; i++) { // Put dim + 2 (22) 1x vectors into the created vectors
            vector<Value> newVec1x;
            this->c_values_4D[layer].back().push_back(newVec1x);
            this->c_values_4D[layer][0].push_back(newVec1x);
            for (unsigned j = 0; j < this->c_dim + 2; j++) {
                this->c_values_4D[layer].back().back().push_back(Value());
                this->c_values_4D[layer].back().back().back().Val = 0.0;
                this->c_values_4D[layer][0].back().push_back(Value());
                this->c_values_4D[layer][0].back().back().Val = 0.0;
            }
        }
    }

    /*for (unsigned layer = 0; layer < 28; layer++) {
        cout << "For layer " << layer << endl;
        for (unsigned i = 0; i < this->c_values_4D[layer].size(); i++) {
            for (unsigned j = 0; j < this->c_values_4D[layer][i].size(); j++) {
                for (unsigned k = 0; k < this->c_values_4D[layer][i][j].size(); k++) {

                    if(this->c_values_4D[layer].size() != 22) {cout << "this->c_values_4D[" << i << "]:         " << endl;}
                    if(this->c_values_4D[layer][i].size() != 22) {cout << "this->c_values_4D[" << i << "][" << j << "]:     " << endl;}
                    if(this->c_values_4D[layer][i][j].size() != 22) {cout << "this->c_values_4D[" << i << "][" << j << "][" << k << "]: " << endl;}
                }
            }
        }
    }*/


    this->c_zero_border = true;
}

void Channel::rzb() // Remove zero border
{
    if (!this->c_zero_border) { cerr << "No zero border to remove" << endl; return;}

    #pragma omp parallel for
    for (unsigned layer = 0; layer < this->c_numLayers; layer++) {
        #pragma omp parallel for
        for (unsigned i = 1; i < this->c_dim + 2; i++) {
            #pragma omp parallel for
            for (unsigned j = 1; j < this->c_dim + 2; j++) {
                this->c_values_4D[layer][i][j].erase(this->c_values_4D[layer][i][j].begin());
                this->c_values_4D[layer][i][j].pop_back();
            }
        }
    }

    #pragma omp parallel for
    for (unsigned layer = 0; layer < this->c_numLayers; layer++) {
        #pragma omp parallel for
        for (unsigned i = 0; i < this->c_dim + 2; i++) {
            this->c_values_4D[layer][i].erase(this->c_values_4D[layer][i].begin());
            this->c_values_4D[layer][i].pop_back();
        }
    }


    #pragma omp parallel for
    for (unsigned layer = 0; layer < this->c_numLayers; layer++) {
        this->c_values_4D[layer].pop_back();
        this->c_values_4D[layer].erase(this->c_values_4D[layer].begin());
    }

    this->c_zero_border = false;

    /*for (unsigned layer = 0; layer < 28; layer++) {
        cout << "For layer " << layer << endl;
        for (unsigned i = 0; i < this->c_values_4D[layer].size(); i++) {
            for (unsigned j = 0; j < this->c_values_4D[layer][i].size(); j++) {
                for (unsigned k = 0; k < this->c_values_4D[layer][i][j].size(); k++) {

                    if(this->c_values_4D[layer].size() != 20) {cout << "this->c_values_4D[" << layer << "]: " << this->c_values_4D[layer].size() << endl;}
                    if(this->c_values_4D[layer][i].size() != 20) {cout << "this->c_values_4D[" << layer << "][" << i << "]: " << this->c_values_4D[layer][i].size() << endl;}
                    if(this->c_values_4D[layer][i][j].size() != 20) {cout << "this->c_values_4D[" << layer << "][" << i << "][" << j << "]: " << this->c_values_4D[layer][i][j].size() << endl;}
                }
            }
        }
    }*/
}

void Channel::clear()
{
    for (unsigned i = 0; i < this->c_numLayers; i++) {
        for (unsigned j = 0; j < this->c_dim; j++) {
            for (unsigned k = 0; k < this->c_dim; k++) {
                for (unsigned l = 0; l < this->c_dim; l++) {
                    this->c_values_4D[i][j][k][l].Val = 0.0;
                    this->c_values_4D[i][j][k][l].deltaVal = 0.0;
                }
            }
        }
    }
}


void Channel::printChannel()
{
    for (unsigned i = 0; i < this->c_numLayers ; i++) {
        cout << "   Channel Layer#" << i << endl;
        for (unsigned j = 0; j < this->c_dim; j++) {
            for (unsigned k = 0; k < this->c_dim; k++) {
                for (unsigned l = 0; l < this->c_dim; l++) {
                    cout << this->c_values_4D[i][j][k][l].Val << " ";
                }
            }
        }
        cout << endl;
    }
}

void Channel::printChannelReduced()
{
    cout << "   Channel Layer#0" << endl;
    for (unsigned j = 0; j < this->c_dim; j++) {
        for (unsigned k = 0; k < this->c_dim; k++) {
            for (unsigned l = 0; l < this->c_dim; l++) {
                cout << this->c_values_4D[0][j][k][l].Val << " ";
            }
        }
    }
}

void Channel::itn()
{
    for (unsigned layer = 0; layer < this->c_numLayers ; layer++) {
        for (unsigned i = 0; i < this->c_dim; i++) {
            for (unsigned j = 0; j < this->c_dim; j++) {
                for (unsigned k = 0; k < this->c_dim; k++) {
                    if(this->c_values_4D[layer][i][j][k].Val != this->c_values_4D[layer][i][j][k].Val) {
                        cout << "NaN found | Layer: " << layer << " i: " << i << " j: " << j << " k: " << k << endl;
                    }
                }
            }
        }
    }
}

long double Channel::sumVals(unsigned layer)
{
    long double sum = 0.0;

    for (unsigned i = 0; i < this->c_dim; i++) {
        for (unsigned j = 0; j < this->c_dim; j++) {
            for (unsigned k = 0; k < this->c_dim; k++) {
                sum += this->c_values_4D[layer][i][j][k].Val;
            }
        }
    }
    return sum;
}
