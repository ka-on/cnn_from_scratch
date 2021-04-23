#ifndef DATAFILE_H
#define DATAFILE_H
#include <fstream>
#include <vector>

using namespace std;

class DataFile
{
    public:
        DataFile(string input, unsigned i, unsigned counter_qgp, unsigned counter_nqgp);
        bool getIs_q();
        void printContents();
        double dat_contents_4D[28][20][20][20];
    private:
        vector<vector<double>> dat_contents_flat;
        bool dat_is_q;
};

#endif // DATAFILE_H
