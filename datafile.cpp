#include "datafile.h"
#include <stdlib.h>

using namespace std;

DataFile::DataFile(string input, unsigned i, unsigned counter_qgp, unsigned counter_nqgp)
{
    string nameoffile;
  if (input == "1") {
      string fileNameInitial = "qgp\\phsd50csr.auau.31.2gev.centr.0000phsd50csr.auau.31.2gev.centr.";
      string fileNameAdd1 = to_string(i);
      string filenameAdd2 = "_event.dat";
      nameoffile = fileNameInitial + fileNameAdd1 + filenameAdd2;
      this->dat_is_q = true;
  } else if (input == "2") {
      string fileNameInitial = "nqgp\\phsd50csr.auau.31.2gev.centr.0000phsd50csr.auau.31.2gev.centr.";
      string fileNameAdd1 = to_string(i);
      string filenameAdd2 = "_event.dat";
      nameoffile = fileNameInitial + fileNameAdd1 + filenameAdd2;
      this->dat_is_q = false;
  } else if (input == "3") {
      int coinFlip = rand() % 2;
      string fileNameInitial = (coinFlip == 1) ? "qgp\\phsd50csr.auau.31.2gev.centr.0000phsd50csr.auau.31.2gev.centr." : "nqgp\\phsd50csr.auau.31.2gev.centr.0000phsd50csr.auau.31.2gev.centr.";
      string fileNameAdd1 = to_string(coinFlip ? counter_qgp : counter_nqgp);
      string filenameAdd2 = "_event.dat";
      nameoffile = fileNameInitial + fileNameAdd1 + filenameAdd2;
      this->dat_is_q = coinFlip ? true : false;
  }
    //////////////////////////////////////////////
    cout << "\nReading file: " << nameoffile << endl;
    char digitFromFile;
    ifstream openFile(nameoffile);
    if(openFile.is_open())
    {
        unsigned particleNum = 0;
        unsigned i = 0;
        unsigned j = 0;
        unsigned k = 0;
        while(openFile.get(digitFromFile))
        {
            if(digitFromFile != 32 && digitFromFile != 10) {
                this->dat_contents_4D[particleNum][i][j][k] = (digitFromFile-48); // * (1.0 / 3.0)
                k += 1;
                if (k == 20) {
                    k = 0;
                    j += 1;
                } if (j == 20) {
                    j = 0;
                    i += 1;
                } if (i == 20) {
                    i = 0;
                }
            } else if (digitFromFile == 10) {
                particleNum += 1;
            }

            /*if(digitFromFile != 32 && digitFromFile != 10){
              this->dat_contents_flat[particleNum].push_back( (digitFromFile-48) ); // * (1.0 / 3.0)
            }

            if(this->dat_contents_flat[particleNum].size() == 8000){particleNum += 1;}*/
        }
    } else {
        cerr << "Error occured while opening file: " << nameoffile << endl;
        cerr << "File couldn't be opened. Make sure the data files are in the folders (qpg or nqgp folders in the project folder)" << endl;
        cerr << "and that the directories are correct." << endl;
        exit (EXIT_FAILURE);
    }
    openFile.close();
}

bool DataFile::getIs_q()
{
    return this->dat_is_q;
}

void DataFile::printContents()
{
    for (unsigned i = 0; i < 28; i++) {
        for (unsigned j = 0; j < 20; j++) {
            for (unsigned k = 0; k < 20; k++) {
                for (unsigned l = 0; l < 20; l++) {
                    cout << this->dat_contents_4D[i][j][k][l];
                }
            }
        }
    }
    cout << endl;
}
