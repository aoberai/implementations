#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <array>
#include <random>
#include <algorithm>
#include "math.h"
#include "utils.hpp"
#include "assert.h"

using namespace std;

// TODO: Make it work for inputs greater than 1d

class Neuron {
	private:
		double value=0; // activationFunction(x * W + b)
		double bias=0; 
		double z=0; // x * W + b
	public:
		Neuron() {
		}

		Neuron(double value, double bias) {
			this->value = value;
			this->bias = bias;
		}

		double GetBias() {
			return this->bias;
		}

		void SetBias(double bias) {
			this->bias = bias;
		}

		void ModifyBias(double delta) {
			this->bias += delta;
		}

		double GetValue() {
			return this->value; }

		void SetValue(double value) {
			this->value = value;	
		}

		double GetZValue() {
			return this->z; }

		void SetZValue(double z) {
			this->z= z;	
		}

		string ToString() {
			return "Neuron( Value: " + to_string(value)  + " Bias: " + to_string(bias) + " ) ";
		}
};

class LossFunctionBase {
	public:
		virtual double Evaluate(vector<double> y, vector<double> outputs) = 0;
		virtual double dCbydA(double y, double values) = 0;
		virtual string ToString() = 0;
};

class MeanSquaredError : public LossFunctionBase {
	public: 
		double Evaluate(vector<double> y, vector<double> outputs) {
			double summation = 0;
			assert (y.size() == outputs.size());
			for (int i = 0; i < y.size(); i++) {
				summation += pow((y[i] - outputs[i]), 2);
			}
			return summation / y.size();
		}

		//  dC/dA = for MSE, (2/n) * summation from 1 to n (ai - yi)
		double dCbydA(double y, double values) {
			return  2 * (values - y);
		}
		string ToString() {
			return "MSE";
		}
};

class CategoricalCrossEntropy : public LossFunctionBase {
	public:
		// wanted values must be one hot encoded
		double Evaluate(vector<double> y, vector<double> outputs) {
			assert (y.size() == outputs.size());
			auto y_1 = find(y.begin(), y.end(), 1);
			assert(y_1 != y.end()); // Assert wanted values (one hot) contains 1
			int y_1_index = y_1 - y.begin();
			return -log(outputs[y_1_index] - utils::EPSILON); // EPSILON subtraction to ensure log(1) doesnt occur
		}
		double dCbydA(double y, double values) {
			return 0;
		}
		string ToString() {
			return "CategoricalCrossEntropy";
		}
};

class ActivationFunctionBase {
	public:
		virtual vector<double> Forward(vector<double> input) = 0;
		virtual double dAbydz(double value) = 0;
		virtual string ToString() = 0;
};

class IdentityActivationFunction: public ActivationFunctionBase { 
	public: 
		vector<double> Forward(vector<double> input) { 
			return input;
		}
		double dAbydz(double value) {
			return 1;
		}
		string ToString() {
			return "None";
		}
};

/**
 * Don't use if want negative outputs in regression problem
 *
 * Problem: Dying relu is problem when neurons in network become inactive and only output 0 since neuron value < 0 before activationFunction
 */
class Relu: public ActivationFunctionBase {
	public: 
		vector<double> Forward(vector<double> input) {
			vector<double> output;
			for (auto v : input) {
				output.push_back((v > 0) ? v : 0);
			}
			return output;
		}
		double dAbydz(double value) {
			return (value > 0) ? 1 : 0;
		}
		string ToString() {
			return "Relu";
		}
};

/**
 * Good old sigmoid, relu better in many cases for reasons below
 *
 * Problem: Vanishing Gradient Problem occurs since derivative function at tails of sigmoid is 0 and squishes input space to between 0 and 1 where information lost at tails (kills gradients). Shouldn't stack many layers with sigmoid as activation function.
 * 					Slow compared to Relu
 *
 * */
class Sigmoid: public ActivationFunctionBase {
	private:
		double Compute(double input) {
			return 1 / (1 + pow(utils::E,-input));
		}
	public: 
		vector<double> Forward(vector<double> input) {
			vector<double> output;
			for (auto v : input) {
				output.push_back(Compute(v));
			}
			return output; 
		}
		double dAbydz(double value) {
			// return 1 / (pow(1 + pow(utils::E, -value), 2)*pow(utils::E, value));
			double sigmoid = Compute(value);
			return sigmoid * (1 - sigmoid);
		}
		string ToString() { 
			return "Sigmoid";
		}
};

/**
 * TanH: TODO implement this 
 *
 * */
class Tanh: public ActivationFunctionBase {
	public: 
		vector<double> Forward(vector<double> input) {
			//TODO:Implement this
			return input;
		}
		double dAbydz(double value) {
			//TODO:Implement this
			return value;
		}
		string ToString() { 
			return "Tanh";
		}
};

/**
 * Used traditionally as last layer of NN. Normalizes outputs to a probability distribution.
 */
class Softmax: public ActivationFunctionBase {
	public: 
		vector<double> Forward(vector<double> input) {
			vector<double> output;
			double summation = 0;
			double max = *max_element(input.begin(), input.end());
			for (auto v : input) {
				output.push_back(pow(utils::E, v - max));
				summation += output.back();
			}
			for (int i = 0; i < output.size(); i++) {
				output[i] = output[i]/summation;
			}

			// Test Code to ensure proper normalization: Should comment out for speed
			// summation = 0;	
			// for (auto v : output) {
			// 	summation += v;
			// }
			// cout << " Summation: " << summation;
			// assert (summation == 1);

			return output; 
		}
		double dAbydz(double value) {
			return 1; // TODO: Implement This
		}
		string ToString() { 
			return "Softmax";
		}
};

class OptimizationAlgorithm {
	public:
		virtual unordered_map<int, map<pair<int, int>, double>> ComputeWeightGradientVector() = 0;
		virtual map<pair<int, int>, double> ComputeBiasGradientVector() = 0;
};

random_device rd;
mt19937 e2(rd());
class Linear {
	private:
		int inFeatures; int outFeatures;
		vector<Neuron*>* neurons = new vector<Neuron*>();
		vector<vector<double>> weights; //outer vector is n layer connection, inner vector is n-1 layer connection
		ActivationFunctionBase* activationFunction;

	public:
		Linear(int inFeatures, int outFeatures) {
			this->activationFunction = new IdentityActivationFunction();
			this->inFeatures = inFeatures;
			this->outFeatures = outFeatures;

			// Initialize Gaussian Random Number Generator
			normal_distribution<> dist(0.5, 0.15); //TODO: change this, too much variability; check xavier initialization; find "goldilocks" zone
			for (int i = 0; i < inFeatures; i++) {
				// (this->neurons)->push_back(new Neuron(dist(e2), 0.1));
				(this->neurons)->push_back(new Neuron());
			}

			for (int i = 0; i < outFeatures; i++) {
				vector<double> tmp(inFeatures, 0);
				for (auto &v : tmp) {
					v = dist(e2);
				}
				weights.push_back(tmp);
			}
		}

		Linear(int inFeatures, int outFeatures, ActivationFunctionBase* activationFunction) : Linear(inFeatures, outFeatures) {
			this->activationFunction = activationFunction;
		}

		vector<Neuron*>* GetNeurons() {
			return neurons;
		}

		vector<double> GetNeuronValues() {
			vector<double> neuron_values;
			for (Neuron* neuron : *neurons) {
				neuron_values.push_back(neuron->GetValue());
			}
			return neuron_values;
		}

		vector<double> GetNeuronsZValues() {
			vector<double> neuron_zvalues;
			for (Neuron* neuron : *neurons) {
				neuron_zvalues.push_back(neuron->GetZValue());
			}
			return neuron_zvalues;
		}

		vector<double> GetNeuronsBias() {
			vector<double> neuron_bias;
			for (Neuron* neuron : *neurons) {
				neuron_bias.push_back(neuron->GetBias());
			}
			return neuron_bias;
		}

		Neuron GetNeuronElement(int n) {
			return *(*neurons)[n];
		}

		// void SetNeuronValue(int n, double value) {
		// 	(*neurons)[n]->SetValue(value);
		// }
    //
		vector<vector<double>> GetWeights() {
			return this->weights;
		}

		void SetElementWeight(int i, int j, double value) {
			this->weights[i][j] = value;
		}

		void ModifyElementWeight(int i, int j, double delta) {
			this->weights[i][j]+=delta;
		}

		string ToString() {
			string output = "Linear ( ";
			for (auto it = (*neurons).begin(); it != (*neurons).end(); ++it) {
				output += (*it)->ToString();
			}

			output += " ) \n\n";
			for (int i = 0; i < this->weights.size(); i++) {
				output += ("Node Weight Input " + to_string(i) + " ");
				for (int j = 0; j < this->weights[i].size(); j++) {
					output += (to_string(this->weights[i][j]) + "  ");
				}
				output += "\n";	
			}

			output += "\n\n";

			output += ("Activation Function: " + this->activationFunction->ToString() + " | Node Count: " + to_string(neurons->size()));
			return output;
		}
		ActivationFunctionBase* GetActivationFunction() {
			return this->activationFunction;
		}
};

class Model {
	private:
		vector<Linear> layers;
		LossFunctionBase* lossFunction = new MeanSquaredError();
		vector<vector<double>> xTrain;
		vector<vector<double>> yTrain;
	public:
		Model() {}

		Model(LossFunctionBase* lossFunction) {
			this->lossFunction = lossFunction;
		}

		Model(initializer_list<Linear> layers_to_add) {
			for (Linear layer : layers_to_add) {
				this->layers.push_back(layer);
			}
		}

		Model(initializer_list<Linear> layers_to_add, LossFunctionBase* lossFunction) {
			this->lossFunction = lossFunction;
			for (Linear layer : layers_to_add) {
				this->layers.push_back(layer);
			}
		}

		void SetLossFunction(LossFunctionBase* lossFunction) {
			this->lossFunction = lossFunction;
		}
		void AddLayer(Linear neurons) {
			this->layers.push_back(neurons);
		}
		Linear GetFirst() {
			return this->layers[0];
		}
		Linear GetLast() {
			return this->layers.back();
		}
		Linear GetElement(int n) {
			return this->layers[n];
		}
		int GetLayerCount() {
			return layers.size();
		}

		vector<double> FeedForward() {
			vector<double> loss;

			for (int c = 0; c < xTrain.size(); c++) {
				assert(xTrain[c].size() == (*(layers[0].GetNeurons())).size());

				for (int j = 0; j < xTrain[c].size(); j++) {
					(*(layers[0].GetNeurons()))[j]->SetValue(xTrain[c][j]);
					(*(layers[0].GetNeurons()))[j]->SetZValue(xTrain[c][j]);
				}

				for (int i = 0; i < layers.size() - 1; i++) {
					vector<vector<double>> layer_weights = layers[i].GetWeights();
					vector<double> inputs = layers[i].GetNeuronValues();
					vector<double> bias = layers[i+1].GetNeuronsBias();
					vector<double> z_outputs; // x * W + b
					vector<double> outputs; // z_outputs after activation function
					for (int j = 0; j < layers[i+1].GetNeurons()->size(); j++) {
						z_outputs.push_back(utils::dot_product(inputs, layer_weights[j]) + bias[i]);
					}

					outputs = layers[i].GetActivationFunction()->Forward(z_outputs);

					for (int j = 0; j < layers[i+1].GetNeurons()->size(); j++) {
						(*(layers[i+1].GetNeurons()))[j]->SetZValue(z_outputs[j]); // Needed for backprop
						(*(layers[i+1].GetNeurons()))[j]->SetValue(outputs[j]);
					}
				}
				loss.push_back(lossFunction->Evaluate(yTrain[c], layers[layers.size() - 1].GetNeuronValues()));
			}
			return loss;
		}

		void FeedForward(vector<double> x) {
			assert(x.size() == (*(layers[0].GetNeurons())).size());
			for (int i = 0; i < x.size(); i++) {
				(*(layers[0].GetNeurons()))[i]->SetValue(x[i]);
				(*(layers[0].GetNeurons()))[i]->SetZValue(x[i]);
			}

			for (int i = 0; i < layers.size() - 1; i++) {
				vector<vector<double>> layer_weights = layers[i].GetWeights();
				vector<double> inputs = layers[i].GetNeuronValues();
				vector<double> bias = layers[i+1].GetNeuronsBias();
				vector<double> z_outputs; // x * W + b
				vector<double> outputs; // z_outputs after activation function
				for (int j = 0; j < layers[i+1].GetNeurons()->size(); j++) {
					z_outputs.push_back(utils::dot_product(inputs, layer_weights[j]) + bias[j]);
				}

				outputs = layers[i].GetActivationFunction()->Forward(z_outputs);

				for (int j = 0; j < layers[i+1].GetNeurons()->size(); j++) {
					(*(layers[i+1].GetNeurons()))[j]->SetZValue(z_outputs[j]); // Needed for backprop
					(*(layers[i+1].GetNeurons()))[j]->SetValue(outputs[j]);
				}
			}
		}

		/**
		double FeedForward(initializer_list<double> x, initializer_list<double> y) {
			assert(x.size() == (*(layers[0].GetNeurons())).size());
			for (int i = 0; i < x.size(); i++) {
				(*(layers[0].GetNeurons()))[i]->SetValue(*(x.begin()+i));
				(*(layers[0].GetNeurons()))[i]->SetZValue(*(x.begin()+i));
			}

			for (int i = 0; i < layers.size() - 1; i++) { 
				vector<vector<double>> layer_weights = layers[i].GetWeights();
				vector<double> inputs = layers[i].GetNeuronValues();
				vector<double> bias = layers[i+1].GetNeuronsBias();
				vector<double> z_outputs; // x * W + b
				vector<double> outputs; // z_outputs after activation function

				for (int j = 0; j < layers[i+1].GetNeurons()->size(); j++) {
					z_outputs.push_back(utils::dot_product(inputs, layer_weights[j]) + bias[i]);
				}
				outputs = layers[i].GetActivationFunction()->Forward(z_outputs);

				for (int j = 0; j < layers[i+1].GetNeurons()->size(); j++) {
					(*(layers[i+1].GetNeurons()))[j]->SetZValue(z_outputs[j]); // Needed for backprop
					(*(layers[i+1].GetNeurons()))[j]->SetValue(outputs[j]);
				}
			}

			double loss = lossFunction->Evaluate(y, layers[layers.size() - 1].GetNeuronValues());
			return loss;
		}

		void FeedForward(double x[], int xLen) {
			assert(xLen == (*(layers[0].GetNeurons())).size());
			for (int i = 0; i < xLen; i++) {
				(*(layers[0].GetNeurons()))[i]->SetValue(x[i]);
				(*(layers[0].GetNeurons()))[i]->SetZValue(x[i]);
			}

			for (int i = 0; i < layers.size() - 1; i++) {
				vector<vector<double>> layer_weights = layers[i].GetWeights();
				vector<double> inputs = layers[i].GetNeuronValues();
				vector<double> bias = layers[i+1].GetNeuronsBias();
				vector<double> z_outputs; // x * W + b
				vector<double> outputs; // z_outputs after activation function
				for (int j = 0; j < layers[i+1].GetNeurons()->size(); j++) {
					z_outputs.push_back(utils::dot_product(inputs, layer_weights[j]) + bias[i]);
				}

				outputs = layers[i].GetActivationFunction()->Forward(z_outputs);

				for (int j = 0; j < layers[i+1].GetNeurons()->size(); j++) {
					(*(layers[i+1].GetNeurons()))[j]->SetZValue(z_outputs[j]); // Needed for backprop
					(*(layers[i+1].GetNeurons()))[j]->SetValue(outputs[j]);
				}
			}
		}


		*/ 


		void BackPropagation(double y[], int yLen) {
		}

		void UpdateWeights(unordered_map<int, map<pair<int, int>, double>> gradientVector) {
			for (unordered_map<int, map<pair<int, int>, double>>::iterator layer = gradientVector.begin(); layer != gradientVector.end(); layer++) {
				for (map<pair<int, int>, double>::iterator it = layer->second.begin(); it != layer->second.end(); it++) {
					// printf("\n(%d, %d) ->  %f\n\n", it->first.first, it->first.second, it->second);
					layers[layer->first].ModifyElementWeight(it->first.first, it->first.second, it->second);
				}
			}
		}

		void UpdateBias(map<pair<int, int>, double> gradientVectorLast) {
			for (map<pair<int, int>, double>::iterator it = gradientVectorLast.begin(); it != gradientVectorLast.end(); it++) {
				// printf("\n\n %d, %d, %0.3f | value: %f", it->first.first, it->first.second, it->second, (*(layers[it->first.first].GetNeurons()))[it->first.second]->GetValue());
				(*(layers[it->first.first].GetNeurons()))[it->first.second]->ModifyBias(it->second);
				// printf("\n\n Applied: %d, %0.3f", it->first.first, (*(layers[layers.size()-1].GetNeurons()))[it->first.second]->GetValue());
				// layers[layers.size() - 1].SetNeuronValue(it->first, it->second);
			}
		}


		LossFunctionBase* GetLossFunction() {
			return this->lossFunction;
		}

		vector<Linear>* GetLayers() {
			return &this->layers;
		}

		void setTrainSet(vector<vector<double>> xTrain, vector<vector<double>> yTrain) {
			this->xTrain = xTrain;
			this->yTrain = yTrain;
		}

		vector<vector<double>> GetXTrain() {
			return this->xTrain;
		}

		vector<vector<double>> GetYTrain() {
			return this->yTrain;
		}

		string ToString() {
			string output = "\n\nModel { \n\n";
			for (Linear layer: layers) {
				output += "    ";
				output += layer.ToString();
				output += "\n\n";
			} 
			output += "} \n\n"; 
			output += "Layer Count: " + to_string(this->layers.size()); 
			return output;
		} 
};


class VanillaGradientDescent : public OptimizationAlgorithm {
		private:
			Model* model;
			double learningRate = 1e-4;
		public: 
			VanillaGradientDescent(Model* model, double learningRate) {
				this->model = model;
				this->learningRate = learningRate;	
			}



			// unordered_map<int, map<pair<int, int>, double>> ComputeWeightGradientVector() {
			// }

			unordered_map<int, map<pair<int, int>, double>> ComputeWeightGradientVector() {
				/*
				 *  What we want to do: compute derivative of cost function with respect to the weights (chain rule) and then add negative of derivative value to all weights in order to adjust weights; which would result in lower cost.
				 * dC / dw = dC/dA * dA / dz * dz / dw 
				 * dz/dw = x dA/dz = given in activation function classes 
				 * dC/dA = for MSE, (2/n) * summation from 1 to n (ai - yi) 
				 * C is cost function, A is activation function, z is (w * x + b)
				 */
				  unordered_map<int, map<pair<int, int>, double>> gradientVector;

					//For now, will just adjust weights of last layer
					vector<Linear> layers = *model->GetLayers();
					LossFunctionBase* lossFunction = model->GetLossFunction();

					vector<double> lastLayerZValues = layers[2].GetNeuronsZValues();
					vector<double> lastLayerValues = layers[2].GetNeuronValues();

					vector<double> secondLastLayerValues = layers[1].GetNeuronValues();

					vector<double> secondLastLayerZValues = layers[1].GetNeuronsZValues();

					vector<double> thirdLastLayerValues = layers[0].GetNeuronValues();

					ActivationFunctionBase* lastLayerActivationFunction = layers[1].GetActivationFunction();	
					ActivationFunctionBase* secondLastLayerActivationFunction = layers[1].GetActivationFunction();	

					vector<vector<double>> yTrain = model->GetYTrain();

					for (int input_index = 0; input_index < yTrain.size(); input_index++) {
						// adjusting last layer weights
						for (int i = 0; i < lastLayerZValues.size(); i++) {
							for (int j = 0; j < secondLastLayerValues.size(); j++) {
								// printf("dCbydA: %f \n", lossFunction->dCbydA(yTrain[input_index][0], lastLayerValues[i]));
								// printf("dAbydz: %f \n", lastLayerActivationFunction->dAbydz(lastLayerZValues[i]));
								// printf("dzbydw: %f \n", secondLastLayerValues[j]);
								double negdCbydzEvaluated = (-1 * lossFunction->dCbydA(yTrain[input_index][0], lastLayerValues[i]) * lastLayerActivationFunction->dAbydz(lastLayerZValues[i]));
								// printf("-dC/dw: %f \n\n\n", negdCbydwEvaluated);
								// adjusting second to last layer weights
								// for (int c = 0; c < secondLastLayerZValues.size(); c++) {
									// for (int k = 0; k < thirdLastLayerValues.size(); k++) {
										// double negdCbydwNMinus1Evaluated = negdCbydzEvaluated * layers[layers.size() - 3].GetWeights()[c][k] * secondLastLayerActivationFunction->dAbydz(secondLastLayerZValues[c]) * thirdLastLayerValues[k];
										// pair<double, double> key = make_pair(c, k);
										// gradientVector[layers.size() - 3][key] += negdCbydwNMinus1Evaluated/yTrain.size();
									// }
								// }
								//
								//
								double negdCbydwEvaluated = negdCbydzEvaluated * secondLastLayerValues[j];
								pair<double, double> key = make_pair(i, j);
								gradientVector[1][key] += negdCbydwEvaluated/yTrain.size();
							}
						}

					}
					return gradientVector;
			}

			map<pair<int, int>, double> ComputeBiasGradientVector() {
					/**
					 *  What we want to do: compute derivative of cost function with respect to the bias (chain rule) and then add negative of derivative value to all biases in order to adjust weights; which would result in lower cost.
					 * dC / db = dC/dA * dA / dz * dz / db 
					 * dz/dw = x 
					 * dA/dz = given in activation function classes 
					 * dz / db = 1
					 * dC/dA = for MSE, (2/n) * summation from 1 to n (ai - yi) 
					 * C is cost function, A is activation function, z is (w * x + b)
					 */
				  map<pair<int, int>, double> gradientVector;

					//For now, will just adjust weights of last layer
					vector<Linear> layers = *model->GetLayers();
					LossFunctionBase* lossFunction = model->GetLossFunction();

					vector<double> lastLayerValues = layers[2].GetNeuronValues();
					vector<double> lastLayerZValues = layers[2].GetNeuronsZValues();
					ActivationFunctionBase* lastLayerActivationFunction = layers[2].GetActivationFunction();	

					vector<vector<double>> yTrain = model->GetYTrain();
					for (int input_index = 0; input_index < yTrain.size(); input_index++) {
						for (int i = 0; i < lastLayerZValues.size(); i++) {
								double negdCbydbEvaluated = (-1 * lossFunction->dCbydA(yTrain[input_index][0], lastLayerValues[i]) * lastLayerActivationFunction->dAbydz(lastLayerZValues[i]));
								// printf("-dC/dw: %f \n\n\n", negdCbydwEvaluated);
								gradientVector[make_pair(2, i)] += negdCbydbEvaluated/yTrain.size();
						}
					}

					return gradientVector;
			}
};


int main() { 
	const int input_shape = 1; 
	const int output_shape = 1;
	const int train_count = 5;
	const int validation_count = 5;
	// Linear layer1 (input_shape, 7, new Sigmoid());
	Linear layer2 (input_shape, 3, new IdentityActivationFunction());
	Linear layer3 (3, output_shape, new Sigmoid());
	Linear layer4 (output_shape, utils::END);

	// (*layer3.GetNeurons())[0]->ModifyBias(5);

	vector<vector<double>> inputs = {{5}, {7}, {6}, {10}, {9}};
	vector<vector<double>> outputs = {{0.7}, {0.7}, {0.7}, {0.7}, {0.7}};

	// TODO: have a set input and output train set inside model class
	// TODO: use optimizer class for gradient calculation and application
	// TODO: Feedforward should only be used for testing outputs

	Model model ({layer2, layer3, layer4});
	model.setTrainSet(inputs, outputs);
	model.SetLossFunction(new MeanSquaredError());
	OptimizationAlgorithm* optimizationAlgorithm = new VanillaGradientDescent(&model, 1e-4);
	double loss = utils::average(model.FeedForward());	
	cout << endl << "Before Backprop Loss: " << loss;

	// 	
	for (int k = 0; k < 50; k++) {
		model.UpdateWeights(optimizationAlgorithm->ComputeWeightGradientVector());
		// model.UpdateBias(optimizationAlgorithm->ComputeBiasGradientVector());
		double loss = utils::average(model.FeedForward());	
		cout << endl << "After Backprop Loss: "<< loss;
	}
  //
	cout << model.ToString();

	vector<vector<double>> validation = {{5}, {7}, {6}, {1}, {2}};
	for (int i = 0; i < 5; i++) {
		model.FeedForward(validation[i]);
		printf("\n\n\n\nInput: %0.1f Output: %0.3f", validation[i][0], model.GetLast().GetNeuronValues()[0]);
	}
	return 0;
}
