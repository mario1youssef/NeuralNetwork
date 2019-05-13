// Network.h: interface for the Network class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_NETWORK_H__8E7C932B_D833_4E1F_9EDC_ED09AFCF876A__INCLUDED_)
#define AFX_NETWORK_H__8E7C932B_D833_4E1F_9EDC_ED09AFCF876A__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

// For FILE
#include <stdio.h>

#include <string>
#include <vector>
#include <fstream>

#define MAX_NET_DIMENSION 25
#define MAX_NET_INPUTS 2		// Cannot exceed the size of the net
#define MAX_NET_OUTPUTS 2		// Cannot exceed the size of the net
#define MAX_DUMMY_STRING_LENGTH 30

class Network
{
public:
	Network();								// default network constructor.
	Network( std::string file_name, bool decider );	// Construct the network from a stored file.
	Network(int inputs, int interneurons, int outputs, char * file_name ); // construct a blank network to user specified size and write it to a file for later editing
	virtual ~Network();

	// Members---------------------
	// Additions to the members should be added to the read and write netork functions

private:
	int numberOfInputs;
	int numberOfOutputs;
	int numberOfInterNeurons;
	int networkDimension;
    int timestep = 0;
    int tStep = 0;

    int numberOfWilsonInputs;
	int numberOfWilsonOutputs;
	int numberOfWilsonInterNeurons;
	int networkWilsonDimension;

	double neuronActivation[MAX_NET_DIMENSION];		// Individual neurons have individual activation levels that are tracked through timesteps an are not visibile as output ( transformed to output by a function)
	double neuronOutput[MAX_NET_DIMENSION];			// The output of individual neurons, used as inputs to the other neurons in the network throught the connection weights matrix
	double neuronThresholds[MAX_NET_DIMENSION];		// Individual Neurons each have a speficifed activation threshold
	double neuronLearningRate[MAX_NET_DIMENSION];	// Individual Neurons each have a speficifed learning rate -- rate of change of connection strength per time step
	short int neuronRefractoryState[MAX_NET_DIMENSION];	// Individual Neurons each have a speficifed period during which output is blocked -- should be 0 or greater.
	double neuronWeightTotal[MAX_NET_DIMENSION];	// Individual Neurons each have a speficifed total weight strength in their input connections.
	double networkWeights[MAX_NET_DIMENSION*MAX_NET_DIMENSION];
	double networkInputs[MAX_NET_INPUTS];
	double networkOutputs[MAX_NET_OUTPUTS];
	short int  plasticWeightsMask[MAX_NET_DIMENSION*MAX_NET_DIMENSION]; // a filter. Plastic weights are = 1, fixed = 0. THis allows for the specification of some fixed and some plastic weights in the same neuron. This could be a binary array ( type bool) to save space.
    FILE* logFile;

    double neuronActivationW [MAX_NET_DIMENSION];		// Individual neurons have individual activation levels that are tracked through timesteps an are not visibile as output ( transformed to output by a function)
	double neuronOutputW [MAX_NET_DIMENSION];			// The output of individual neurons, used as inputs to the other neurons in the network throught the connection weights matrix
	double neuronThresholdsW [MAX_NET_DIMENSION];		// Individual Neurons each have a speficifed activation threshold
	double neuronLearningRateW [MAX_NET_DIMENSION];	// Individual Neurons each have a speficifed learning rate -- rate of change of connection strength per time step
	short int neuronRefractoryStateW [MAX_NET_DIMENSION];	// Individual Neurons each have a speficifed period during which output is blocked -- should be 0 or greater.
	double neuronWeightTotalW [MAX_NET_DIMENSION];	// Individual Neurons each have a speficifed total weight strength in their input connections.
	double networkWeightsW [MAX_NET_DIMENSION*MAX_NET_DIMENSION];
	double networkInputsW [MAX_NET_INPUTS];
	double networkOutputsW [MAX_NET_OUTPUTS];
	short int  plasticWeightsMaskW [MAX_NET_DIMENSION*MAX_NET_DIMENSION];
    FILE* motorOutputFile;

	// Functions -------------------------
	bool openLogFile();
	bool openMotorOutputFile();
	void closeLogFile();
	void closeMotorOutputFile();

	void instantiateDefaultNetwork( void );
	void setNetworkOuput( void );

    void setNetworkOuputW( void );                           //WILSON COPY
	void copyNeuronActivationsToNeuronOutputs( void );

    void copyNeuronActivationsToNeuronOutputsW( void );      //WISLON COPY
	void copyNetworkInputsToInputNeuronOutputs( void );
	void thresholdNeuronOutputs( void );
	void wilsonBoundNetworkOutputs(void);
	void wilsonActivation(void);

	void copyWilsonOutputsToInputNeuronsOutput(void);

	void squashNeuronOutputs( double offset, double expSlope);
	void setNeuronOutput( double value );
	void setNeuronThresholds( double value );
	void setNeuronLearningRate( double value );
	int setNeuronRefractoryState( int value );
	void setPlasticWeightsMask( short int value ); // in general it is good to set this to 1 and let the learning rate determine plasticity.  This is to be used for special cases
	void setNeuronActivation( double value );
	void setNetworkOutputs( double value );
	void networkActivation( void  );
	void hebbianWeightUpdate( void  );
	void hebbianExcitatoryWeightUpdate( void );
	void hebbianInhibitoryWeightUpdate( void );
	void normalizeNeuronWeights( void );			// Update weight totals to neuron-specific values
	void normalizeNeuronWeights( double value );	// Uptdate weight totals to specificed values
	void normalizeNonDiagonalNeuronWeights( void );
	void normalizeNonDiagonalInhibitoryNeuronWeights( void );
	void normalizeNonDiagonalExcitatoryNeuronWeights( void );
	void setNeuronWeightTotal( double value);
	int computeWeightIndex( int source_neuron_number, int target_neuron_number );
	void readRowFromFile(FILE* fp, short* array);
	void readMultipleRowsFromFile(FILE* fp, short* array);
	void readRowFromFile(FILE* fp, double* array);
	void readMultipleRowsFromFile(FILE* fp, double* array);

	void readRowFromFileW(FILE* fp, short* array);
	void readMultipleRowsFromFileW(FILE* fp, short* array);
	void readRowFromFileW(FILE* fp, double* array);
	void readMultipleRowsFromFileW(FILE* fp, double* array);
    int computeWeightIndexW( int source_neuron_number, int target_neuron_number );

    double sigmoid_function(double x);



public:
    void wilsonCycleNetwork(void);
    void cycleNetwork( void );
	void cycleNetworkSquash(  double offset, double expSlope );
	void cycleNetworkNormalizeHebbianLearning( void );
	void printNetworkOuput( void );
	void printNetworkOutputState( void );
	void setNetworkWeightsDiagonalRange( double value, int start_row_col, int end_row_col );
	void setNetworkWeightsUpperLowerTriangleAndDiagonal( double diagonal_value, double upper_triangle_value, double lower_triangle_value);
	void setNetworkWeightsRectangle( double value, int start_row, int end_row, int start_column, int end_column );
	void setNetworkWeightsUpperTriangle( double value, int start_row, int end_row, int start_column, int end_column );
	void setNetworkWeightsLowerTriangle( double value, int start_row, int end_row, int start_column, int end_column );
	void writeNetworkInputToFile( std::string file_name );
	void writeNetworkOutputStateToFile( std::string file_name );
	void writeNetworkActivationStateToFile( std::string file_name );
	void writeNetworkWeightsToFile( std::string file_name );
	void setNetworkInput( double *vector);
	void setNetworkInput(std::vector<double>);
	//void getNetworkOuput( double* vector );
	int readNetworkFromFile( std::string file_name);
	int writeNetworkToFile( std::string file_name );
	double*getNetworkWeights();
	void setNetworkWeights( double value );
	void PrintNetworkState( void);
	void printNetworkWeights();


    void getNetworkOuputW( double* array );

	int getNetworkDimension();
	int getNumInputs();
	int getNumOutputs();
	FILE* getLogFile();
    FILE* getMotorOutputFile();

    int readNetworkFromFileW( std::string file_name);
    int writeNetworkToFileW( std::string file_name );

	/*
	 * Update the weight between two neurons to the given value. Neurons are 1-indexed.
	 */
	void updateWeight(int from_neuron, int to_neuron, double new_weight);

	void printStuff();
	/*
	 * Reset the stored outputs of all neurons.
	 */
	void resetNeuronOutputs();
};

///////////////////////////////////////////////////
// Helper functions
///////////////////////////////////////////////////

/*
 * Return a copy of the given matrix.
 */
double* copyMatrix(double* matrix, int dimension);

/*
 * Print the difference between the beforeMatrix and the afterMatrix.
 */
void printDifferences(double* beforeMatrix, double* afterMatrix, int dimension);

#endif // !defined(AFX_NETWORK_H__8E7C932B_D833_4E1F_9EDC_ED09AFCF876A__INCLUDED_)
