// SIngleMatrixNetwork.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Network.h"
#include <math.h>
#include <stdio.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <iterator>                     //NEW: NEEDED TO USE ITERATOR
#include <algorithm>
using namespace std;

//void doLearning(Network fred, string input_file_name, string prefix);

/*
 * Convert values to strings. Needed because std::to_string is apparently missing
 * from <string>.
 */
template <typename T>
std::string to_string(T value)
{
	//create an output string stream
	ostringstream os ;

	//throw the value into the string stream
	os << value ;

	//convert the string stream into a string and return
	return os.str() ;
}
/*
 * Usage: SingleMatrixNetwork <config_file> <input_file> <weight_directory>
 */
int main(int argc, char* argv[])
{
    double nilly[5];
    ;
    Network Willy("Wilson.txt", false);
    //Willy.writeNetworkToFileW("WILSON2.txt");
    Network legController("ganglia7.txt", true);
    for(int i = 0; i< 16; i++)
    {
        Willy.wilsonCycleNetwork();
        Willy.getNetworkOuputW(nilly);
        legController.setNetworkInput(nilly);
        legController.cycleNetwork();

    }
    //legController.writeNetworkToFile("ganglia70.txt");
	return 0;
}

