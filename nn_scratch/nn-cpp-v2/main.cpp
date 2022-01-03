#include <iostream>
#include <vector>
#include "utils.hpp"
#include <random>

using namespace std;

random_device rd;                                                                                                                                                
mt19937 e2(rd());
// TODO: Find better initialization method
normal_distribution<> dist(0, 1);

class Neuron {
	public:
		Neuron() {
			this->value = dist(e2);
			this->bias = 0;
		}
		Neuron(double value, double bias) {
			this->value = value;
			this->bias = bias;
		}
		double GetBias() {
			return this->bias;
		}
		double GetValue() {
			return this->value;
		}
		void SetValue(double value) {
			this->value = value;
		}
		void SetBias(double bias) {
			this->bias = bias;
		}
	private:
		double value = 0;
		double bias = 0;
};

class LossFunctionBase {
	public:
		virtual double Evaluate(vector<double> outputs, vector<double> groundTruth) = 0;
		virtual vector<double> Gradient() = 0;
		virtual string ToString() = 0;
};

/*
 * aka L2 loss
 */
class MeanSquaredError : public LossFunctionBase {
	public:
		double Evaluate(vector<double> outputs, vector<double> groundTruth) {
			this->outputs = outputs;
			double summation = 0;
			for (int i = 0; i < outputs.size(); i++) {
				summation += ((outputs[i] - groundTruth[i]) * (outputs[i] - groundTruth[i]));
			}
			return summation/outputs.size();
		}
		vector<double> Gradient() {
			vector<double> gradient;
			//TODO: implement this; not sure how to take derivative across average function. 
			return gradient;
		}
	private:
		vector<double> outputs;
};

class LayerBase {
	public:
		virtual vector<double> Forward(vector<double> inputs) = 0;
		virtual vector<double> Gradient() = 0;
		virtual string ToString() = 0;
};

class ReluActivation : public LayerBase {
	public: 
		vector<double> Forward(vector<double> input) { 
			vector<double> output;                                                                                                                                            
			for (auto v : input) { 
				output.push_back((v > 0) ? v : 0);
			}
			this->input = input;
			return output;
		}                                                                                                                                                                               
		// Derivative of activation function w.r.t z aka (w * input + b)
		vector<double> Gradient() { 
			vector<double> gradient;
			for (int i = 0; i < input.size(); i++) {
				gradient.push_back((input[i] > 0) ? 1 : 0);
			}
			return gradient;
		}
		string ToString() { 
			return "Relu";
		}
	private:
		vector<double> input;
};

class IdentityActivation : public LayerBase {
	public:
		IdentityActivation() {
		}
		vector<double> Forward(vector<double> input) {
			this->inputSize = input.size();
			return input;
		}
		// Derivative of activation function w.r.t z aka (w * input + b)
		vector<double> Gradient() {
			return vector<double>(inputSize, 1);
		}
		string ToString() {
			return "IdentityActivationFunction";
		}
	private:
		int inputSize = 0;
};

class Linear : public LayerBase {
	public:
		Linear(int inChannel, int outChannel) {
			this->inChannel = inChannel;
			this->outChannel = outChannel;
		  for (int i = 0; i < outChannel; i++) {
				vector<double> weightSet;
				for (int j = 0; j < inChannel; j++) {
					weightSet.push_back(dist(e2));
				}
				weights.push_back(weightSet);
				neurons.push_back(Neuron());
			}
			// Weight Visualization
			// for (vector<double> weight : weights) {
			// 	cout << utils::ToString(weight) << endl;
			// }
		}
		vector<double> Forward(vector<double> inputs) {
			assert(inputs.size() == inChannel);
			vector<double> outputs;
			cout << "\n\nLayer: ";
			for (int i = 0; i < this->outChannel; i++) {
				cout << "\n\nInputs: " << utils::ToString(inputs) << "\n";
				cout << "Weights : " << utils::ToString(weights[i]) << "\n";
				outputs.push_back(utils::Dot(inputs, weights[i]) + neurons[i].GetBias());
				cout << "Outputs: " << utils::ToString(outputs);
			}
			// Updates neuron values
			for (int i = 0; i < neurons.size(); i++) {
				neurons[i].SetValue(outputs[i]);
			}
			this->outputs = outputs;
			return outputs;
		}
		// Derivative of z  aka (w * input + b) w.r.t weight
		vector<double> Gradient() {
			return outputs; 
		}
		string ToString() {
			return "Linear";
		}
	private:
		int inChannel, outChannel;
		vector<Neuron> neurons;
		vector<double> outputs;
		vector<vector<double>> weights;
};

class NNet {
	public:
		virtual vector<double> Forward(vector<double> inputs) = 0;
		virtual void BackProp() = 0;
		virtual string Summary() = 0;
};

/*
 * Like TF Sequential API
 */
class LinRegNet : public NNet {
	public:
		LinRegNet() {
			// Build NN architecure here
			vector<LayerBase*> net {new Linear(1, 5), new IdentityActivation(), new Linear(5, 1), new IdentityActivation()};
			this->net = net;
			// net = {new Linear(1, 5)};
		}
		vector<double> Forward(vector<double> inputs) {
			// Forward Prop
			vector<double> nextLayerInput = inputs;
			for (LayerBase* layer : net) {
				nextLayerInput = layer->Forward(nextLayerInput);
			}
			return nextLayerInput;
		}
		void BackProp() {
			//TODO: implement this
			// cout << "\n\n\nStarting BackProp \n";
			// vector<double> dCbydW = ;
			// for (int i = net.size() - 1; i >= 0; --i) {
      //
			// 	dCbydW *= net[i]->Gradient();
			// }
		}
		string Summary() {
			return "LinRegNet";
		}
	private:
		vector<LayerBase*> net;
};
/*
 * Like TF Functional API
 */
class LinRegNet2 : public NNet {
	public:
		vector<double> Forward(vector<double> inputs) {
			// Forward Prop
			vector<double> nextLayerInput = inputs;
			for (LayerBase* layer : net) {
				nextLayerInput = layer->Forward(nextLayerInput);
			}
			return nextLayerInput;
		}
		void BackProp() {
			//TODO: implement this
			// cout << "\n\n\nStarting BackProp \n";
			// vector<double> dCbydW = ;
			// for (int i = net.size() - 1; i >= 0; --i) {
      //
			// 	dCbydW *= net[i]->Gradient();
			// }
		}
		string Summary() {
			return "LinRegNet";
		}
	private:
		Linear d1(1, 5);
		IdentityActivation a1;
		Linear d2(5, 1);
		IdentityActivation a2;

		vector<LayerBase*> net;
};
int main () {
	LinRegNet nnet;
	vector<double> inputs = {1};
	vector<double> outputs = nnet.Forward(inputs);
	nnet.BackProp();
	return 0;
}
