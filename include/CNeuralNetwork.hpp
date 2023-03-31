/*!
 * \file CNeuralNetwork.hpp
 * \brief Declaration of the neural network class
 * \author E.C.Bunschoten
 * \version 1.0.0
 */

#pragma once

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>

#include "CLayer.hpp"
#include "variable_def.hpp"

namespace MLPToolbox {
class CNeuralNetwork {
  /*!
   *\class CNeuralNetwork
   *\brief The CNeuralNetwork class allows for the evaluation of a loaded MLP
   *architecture for a given set of inputs. The class also contains a list of
   *the various supported activation function types (linear, relu, elu, gelu,
   *selu, sigmoid, swish, tanh, exp)which can be applied to the layers in the
   *network. Currently, only dense, feed-forward type neural nets are supported
   *in this implementation.
   */
private:
  std::vector<std::string> input_names, /*!< MLP input variable names. */
      output_names;                     /*!< MLP output variable names. */

  unsigned long n_hidden_layers = 0; /*!< Number of hidden layers (layers
                                        between input and output layer). */

  CLayer *inputLayer = nullptr, /*!< Pointer to network input layer. */
      *outputLayer = nullptr;   /*!< Pointer to network output layer. */

  std::vector<CLayer *> hiddenLayers; /*!< Hidden layer collection. */
  std::vector<CLayer *>
      total_layers; /*!< Hidden layers plus in/output layers */

  // std::vector<su2activematrix> weights_mat; /*!< Weights of synapses
  // connecting layers */
  std::vector<std::vector<std::vector<mlpdouble>>> weights_mat;

  std::vector<std::pair<mlpdouble, mlpdouble>>
      input_norm,  /*!< Normalization factors for network inputs */
      output_norm; /*!< Normalization factors for network outputs */

  std::vector<mlpdouble> last_inputs; /*!< Inputs from previous lookup operation.
                                      Evaluation of the network */
  /*!< is skipped if current inputs are the same as the last inputs. */

  mlpdouble *ANN_outputs; /*!< Pointer to network outputs */
  std::vector<std::vector<mlpdouble>>
      dOutputs_dInputs; /*!< Network output derivatives w.r.t inputs */

  /*!
   * \brief Available activation function enumeration.
   */
  enum class ENUM_ACTIVATION_FUNCTION {
    NONE = 0,
    LINEAR = 1,
    RELU = 2,
    ELU = 3,
    GELU = 4,
    SELU = 5,
    SIGMOID = 6,
    SWISH = 7,
    TANH = 8,
    EXPONENTIAL = 9
  };

  /*!
   * \brief Available activation function map.
   */
  std::map<std::string, ENUM_ACTIVATION_FUNCTION> activation_function_map{
      {"none", ENUM_ACTIVATION_FUNCTION::NONE},
      {"linear", ENUM_ACTIVATION_FUNCTION::LINEAR},
      {"elu", ENUM_ACTIVATION_FUNCTION::ELU},
      {"relu", ENUM_ACTIVATION_FUNCTION::RELU},
      {"gelu", ENUM_ACTIVATION_FUNCTION::GELU},
      {"selu", ENUM_ACTIVATION_FUNCTION::SELU},
      {"sigmoid", ENUM_ACTIVATION_FUNCTION::SIGMOID},
      {"swish", ENUM_ACTIVATION_FUNCTION::SWISH},
      {"tanh", ENUM_ACTIVATION_FUNCTION::TANH},
      {"exponential", ENUM_ACTIVATION_FUNCTION::EXPONENTIAL}};

  std::vector<ENUM_ACTIVATION_FUNCTION>
      activation_function_types; /*!< Activation function type for each layer in
                                    the network. */
  std::vector<std::string>
      activation_function_names; /*!< Activation function name for each layer in
                                    the network. */

public:
  ~CNeuralNetwork() {
    delete inputLayer;
    delete outputLayer;
    for (std::size_t i = 1; i < total_layers.size() - 1; i++) {
      delete total_layers[i];
    }
    delete[] ANN_outputs;
  };
  /*!
   * \brief Set the input layer of the network.
   * \param[in] n_neurons - Number of inputs
   */
  void DefineInputLayer(unsigned long n_neurons) {
    /*--- Define the input layer of the network ---*/
    inputLayer = new CLayer(n_neurons);

    /* Mark layer as input layer */
    inputLayer->SetInput(true);
    input_norm.resize(n_neurons);
    input_names.resize(n_neurons);
  }

  /*!
   * \brief Set the output layer of the network.
   * \param[in] n_neurons - Number of outputs
   */
  void DefineOutputLayer(unsigned long n_neurons) {
    /*--- Define the output layer of the network ---*/
    outputLayer = new CLayer(n_neurons);
    output_norm.resize(n_neurons);
    output_names.resize(n_neurons);
  }

  /*!
   * \brief Add a hidden layer to the network
   * \param[in] n_neurons - Hidden layer size.
   */
  void PushHiddenLayer(unsigned long n_neurons) {
    /*--- Add a hidden layer to the network ---*/
    CLayer *newLayer = new CLayer(n_neurons);
    hiddenLayers.push_back(newLayer);
    n_hidden_layers++;
  }

  /*!
   * \brief Set the weight value of a specific synapse.
   * \param[in] i_layer - current layer.
   * \param[in] i_neuron - neuron index in current layer.
   * \param[in] j_neuron - neuron index of connecting neuron.
   * \param[in] value - weight value.
   */
  void SetWeight(unsigned long i_layer, unsigned long i_neuron,
                 unsigned long j_neuron, mlpdouble value) {
    weights_mat[i_layer][j_neuron][i_neuron] = value;
  };

  /*!
   * \brief Set bias value at a specific neuron.
   * \param[in] i_layer - Layer index.
   * \param[in] i_neuron - Neuron index of current layer.
   * \param[in] value - Bias value.
   */
  void SetBias(unsigned long i_layer, unsigned long i_neuron, mlpdouble value) {
    total_layers[i_layer]->SetBias(i_neuron, value);
  }

  /*!
   * \brief Set layer activation function.
   * \param[in] i_layer - Layer index.
   * \param[in] input - Activation function name.
   */
  void SetActivationFunction(unsigned long i_layer, std::string input) {
    /*--- Translate activation function name from input file to a number ---*/

    activation_function_names[i_layer] = input;

    // Set activation function type in current layer.
    activation_function_types[i_layer] =
        activation_function_map.find(input)->second;

    return;
  }


  /*!
   * \brief Display the network architecture in the terminal.
   */
  void DisplayNetwork() const {
    /*--- Display information on the MLP architecture ---*/
    int display_width = 54;
    int column_width = int(display_width / 3.0) - 1;

    /*--- Input layer information ---*/
    cout << "+" << setfill('-') << setw(display_width) << right << "+" << endl;
    cout << setfill(' ');
    cout << "|" << left << setw(display_width - 1) << "Input Layer Information:"
        << "|" << endl;
    cout << "+" << setfill('-') << setw(display_width) << right << "+" << endl;
    cout << setfill(' ');
    cout << "|" << left << setw(column_width) << "Input Variable:"
        << "|" << left << setw(column_width) << "Lower limit:"
        << "|" << left << setw(column_width) << "Upper limit:"
        << "|" << endl;
    cout << "+" << setfill('-') << setw(display_width) << right << "+" << endl;
    cout << setfill(' ');

    /*--- Hidden layer information ---*/
    for (auto iInput = 0u; iInput < inputLayer->GetNNeurons(); iInput++)
      cout << "|" << left << setw(column_width)
          << to_string(iInput + 1) + ": " + input_names[iInput] << "|" << right
          << setw(column_width) << input_norm[iInput].first << "|" << right
          << setw(column_width) << input_norm[iInput].second << "|" << endl;
    cout << "+" << setfill('-') << setw(display_width) << right << "+" << endl;
    cout << setfill(' ');
    cout << "|" << left << setw(display_width - 1) << "Hidden Layers Information:"
        << "|" << endl;
    cout << "+" << setfill('-') << setw(display_width) << right << "+" << endl;
    cout << setfill(' ');
    cout << "|" << setw(column_width) << left << "Layer index"
        << "|" << setw(column_width) << left << "Neuron count"
        << "|" << setw(column_width) << left << "Function"
        << "|" << endl;
    cout << "+" << setfill('-') << setw(display_width) << right << "+" << endl;
    cout << setfill(' ');
    for (auto iLayer = 0u; iLayer < n_hidden_layers; iLayer++)
      cout << "|" << setw(column_width) << right << iLayer + 1 << "|"
          << setw(column_width) << right << hiddenLayers[iLayer]->GetNNeurons()
          << "|" << setw(column_width) << right
          << activation_function_names[iLayer + 1] << "|" << endl;
    cout << "+" << setfill('-') << setw(display_width) << right << "+" << endl;
    cout << setfill(' ');

    /*--- Output layer information ---*/
    cout << "|" << left << setw(display_width - 1) << "Output Layer Information:"
        << "|" << endl;
    cout << "+" << setfill('-') << setw(display_width) << right << "+" << endl;
    cout << setfill(' ');
    cout << "|" << left << setw(column_width) << "Output Variable:"
        << "|" << left << setw(column_width) << "Lower limit:"
        << "|" << left << setw(column_width) << "Upper limit:"
        << "|" << endl;
    cout << "+" << setfill('-') << setw(display_width) << right << "+" << endl;
    cout << setfill(' ');
    for (auto iOutput = 0u; iOutput < outputLayer->GetNNeurons(); iOutput++)
      cout << "|" << left << setw(column_width)
          << to_string(iOutput + 1) + ": " + output_names[iOutput] << "|"
          << right << setw(column_width) << output_norm[iOutput].first << "|"
          << right << setw(column_width) << output_norm[iOutput].second << "|"
          << endl;
    cout << "+" << setfill('-') << setw(display_width) << right << "+" << endl;
    cout << setfill(' ');
    cout << endl;
  }

  /*!
   * \brief Size the weight layers in the network according to its architecture.
   */
  void SizeWeights() {
    /*--- Size weight matrices based on neuron counts in each layer ---*/

    /* Generate std::vector containing input, output, and hidden layer references
    */
    total_layers.resize(n_hidden_layers + 2);
    total_layers[0] = inputLayer;
    for (auto iLayer = 0u; iLayer < n_hidden_layers; iLayer++) {
      total_layers[iLayer + 1] = hiddenLayers[iLayer];
    }
    total_layers[total_layers.size() - 1] = outputLayer;

    weights_mat.resize(n_hidden_layers + 1);
    weights_mat[0].resize(hiddenLayers[0]->GetNNeurons());
    for (auto iNeuron = 0u; iNeuron < hiddenLayers[0]->GetNNeurons(); iNeuron++)
      weights_mat[0][iNeuron].resize(inputLayer->GetNNeurons());

    for (auto iLayer = 1u; iLayer < n_hidden_layers; iLayer++) {
      weights_mat[iLayer].resize(hiddenLayers[iLayer]->GetNNeurons());
      for (auto iNeuron = 0u; iNeuron < hiddenLayers[iLayer]->GetNNeurons();
          iNeuron++) {
        weights_mat[iLayer][iNeuron].resize(
            hiddenLayers[iLayer - 1]->GetNNeurons());
      }
    }
    weights_mat[n_hidden_layers].resize(outputLayer->GetNNeurons());
    for (auto iNeuron = 0u; iNeuron < outputLayer->GetNNeurons(); iNeuron++) {
      weights_mat[n_hidden_layers][iNeuron].resize(
          hiddenLayers[n_hidden_layers - 1]->GetNNeurons());
    }

    ANN_outputs = new mlpdouble[outputLayer->GetNNeurons()];
    dOutputs_dInputs.resize(outputLayer->GetNNeurons());
    for (auto iOutput = 0u; iOutput < outputLayer->GetNNeurons(); iOutput++)
      dOutputs_dInputs[iOutput].resize(inputLayer->GetNNeurons());

    for (auto iLayer = 0u; iLayer < n_hidden_layers + 2; iLayer++) {
      total_layers[iLayer]->SizeGradients(inputLayer->GetNNeurons());
    }
  }

  /*!
   * \brief Size the std::vector of previous inputs.
   * \param[in] n_inputs - Number of inputs.
   */
  void SizeInputs(unsigned long n_inputs) {
    last_inputs.resize(n_inputs);
    for (unsigned long iInput = 0; iInput < n_inputs; iInput++)
      last_inputs[iInput] = 0.0;
  }

  /*!
   * \brief Get the number of connecting regions in the network.
   * \returns number of spaces in between layers.
   */
  unsigned long GetNWeightLayers() const { return total_layers.size() - 1; }

  /*!
   * \brief Get the total number of layers in the network
   * \returns number of netowork layers.
   */
  unsigned long GetNLayers() const { return total_layers.size(); }

  /*!
   * \brief Get neuron count in a layer.
   * \param[in] iLayer - Layer index.
   * \returns number of neurons in the layer.
   */
  unsigned long GetNNeurons(unsigned long iLayer) const {
    return total_layers[iLayer]->GetNNeurons();
  }

  /*!
   * \brief Evaluate the network.
   * \param[in] inputs - Network input variable values.
   * \param[in] compute_gradient - Compute the derivatives of the outputs wrt
   * inputs.
   */
  void Predict(std::vector<mlpdouble> &inputs, bool compute_gradient = false) {
    /*--- Evaluate MLP for given inputs ---*/

    mlpdouble y = 0, dy_dx = 0; // Activation function output.
    bool same_point = true;
    /* Normalize input and check if inputs are the same w.r.t last evaluation */
    for (auto iNeuron = 0u; iNeuron < inputLayer->GetNNeurons(); iNeuron++) {
      mlpdouble x_norm = (inputs[iNeuron] - input_norm[iNeuron].first) /
                      (input_norm[iNeuron].second - input_norm[iNeuron].first);
      if (abs(x_norm - inputLayer->GetOutput(iNeuron)) > 0)
        same_point = false;
      inputLayer->SetOutput(iNeuron, x_norm);

      if (compute_gradient)
        inputLayer->SetdYdX(
            iNeuron, iNeuron,
            1 / (input_norm[iNeuron].second - input_norm[iNeuron].first));
    }
    /* Skip evaluation process if current point is the same as during the previous
    * evaluation */
    if (!same_point) {
      mlpdouble alpha = 1.67326324;
      mlpdouble lambda = 1.05070098;
      /* Traverse MLP and compute inputs and outputs for the neurons in each layer
      */
      for (auto iLayer = 1u; iLayer < n_hidden_layers + 2; iLayer++) {
        auto nNeurons_current =
            total_layers[iLayer]->GetNNeurons(); // Neuron count of current layer
        mlpdouble x;                                // Neuron input value

        /* Compute and store input value for each neuron */
        for (auto iNeuron = 0u; iNeuron < nNeurons_current; iNeuron++) {
          x = ComputeX(iLayer, iNeuron);
          total_layers[iLayer]->SetInput(iNeuron, x);
          if (compute_gradient) {
            for (auto iInput = 0u; iInput < inputLayer->GetNNeurons(); iInput++) {
              dy_dx = ComputedOutputdInput(iLayer, iNeuron, iInput);
              total_layers[iLayer]->SetdYdX(iNeuron, iInput, dy_dx);
            }
          }
        }

        /* Compute and store neuron output based on activation function */
        switch (activation_function_types[iLayer]) {
        case ENUM_ACTIVATION_FUNCTION::ELU:
          for (auto iNeuron = 0u; iNeuron < nNeurons_current; iNeuron++) {
            x = total_layers[iLayer]->GetInput(iNeuron);
            if (x > 0) {
              y = x;
              if (compute_gradient)
                dy_dx = 1.0;
            } else {
              y = exp(x) - 1;
              if (compute_gradient)
                dy_dx = exp(x);
            }
            total_layers[iLayer]->SetOutput(iNeuron, y);
            if (compute_gradient) {
              for (auto iInput = 0u; iInput < inputLayer->GetNNeurons();
                  iInput++) {
                total_layers[iLayer]->SetdYdX(
                    iNeuron, iInput,
                    dy_dx * total_layers[iLayer]->GetdYdX(iNeuron, iInput));
              }
            }
          }
          break;
        case ENUM_ACTIVATION_FUNCTION::LINEAR:
          for (auto iNeuron = 0u; iNeuron < nNeurons_current; iNeuron++) {
            x = total_layers[iLayer]->GetInput(iNeuron);
            y = x;
            total_layers[iLayer]->SetOutput(iNeuron, y);
            if (compute_gradient) {
              dy_dx = 1.0;
              for (auto iInput = 0u; iInput < inputLayer->GetNNeurons();
                  iInput++) {
                total_layers[iLayer]->SetdYdX(
                    iNeuron, iInput,
                    dy_dx * total_layers[iLayer]->GetdYdX(iNeuron, iInput));
              }
            }
          }
          break;
        case ENUM_ACTIVATION_FUNCTION::EXPONENTIAL:
          for (auto iNeuron = 0u; iNeuron < nNeurons_current; iNeuron++) {
            x = total_layers[iLayer]->GetInput(iNeuron);
            y = exp(x);
            total_layers[iLayer]->SetOutput(iNeuron, y);
            if (compute_gradient) {
              dy_dx = 1.0;
              for (auto iInput = 0u; iInput < inputLayer->GetNNeurons();
                  iInput++) {
                total_layers[iLayer]->SetdYdX(
                    iNeuron, iInput,
                    dy_dx * total_layers[iLayer]->GetdYdX(iNeuron, iInput));
              }
            }
          }
          break;
        case ENUM_ACTIVATION_FUNCTION::RELU:
          for (auto iNeuron = 0u; iNeuron < nNeurons_current; iNeuron++) {
            x = total_layers[iLayer]->GetInput(iNeuron);
            if (x > 0) {
              y = x;
              if (compute_gradient)
                dy_dx = 1.0;
            } else {
              y = 0.0;
              if (compute_gradient)
                dy_dx = 0.0;
            }
            total_layers[iLayer]->SetOutput(iNeuron, y);
            if (compute_gradient) {
              for (auto iInput = 0u; iInput < inputLayer->GetNNeurons();
                  iInput++) {
                total_layers[iLayer]->SetdYdX(
                    iNeuron, iInput,
                    dy_dx * total_layers[iLayer]->GetdYdX(iNeuron, iInput));
              }
            }
          }
          break;
        case ENUM_ACTIVATION_FUNCTION::SWISH:
          for (auto iNeuron = 0u; iNeuron < nNeurons_current; iNeuron++) {
            x = total_layers[iLayer]->GetInput(iNeuron);
            y = x / (1 + exp(-x));
            total_layers[iLayer]->SetOutput(iNeuron, y);
            if (compute_gradient) {
              dy_dx = exp(x) * (x + exp(x) + 1) / pow(exp(x) + 1, 2);
              for (auto iInput = 0u; iInput < inputLayer->GetNNeurons();
                  iInput++) {
                total_layers[iLayer]->SetdYdX(
                    iNeuron, iInput,
                    dy_dx * total_layers[iLayer]->GetdYdX(iNeuron, iInput));
              }
            }
          }
          break;
        case ENUM_ACTIVATION_FUNCTION::TANH:
          for (auto iNeuron = 0u; iNeuron < nNeurons_current; iNeuron++) {
            x = total_layers[iLayer]->GetInput(iNeuron);
            y = tanh(x);
            total_layers[iLayer]->SetOutput(iNeuron, y);
            if (compute_gradient) {
              dy_dx = 1 / pow(cosh(x), 2);
              for (auto iInput = 0u; iInput < inputLayer->GetNNeurons();
                  iInput++) {
                total_layers[iLayer]->SetdYdX(
                    iNeuron, iInput,
                    dy_dx * total_layers[iLayer]->GetdYdX(iNeuron, iInput));
              }
            }
          }
          break;
        case ENUM_ACTIVATION_FUNCTION::SIGMOID:
          for (auto iNeuron = 0u; iNeuron < nNeurons_current; iNeuron++) {
            x = total_layers[iLayer]->GetInput(iNeuron);
            y = 1.0 / (1 + exp(-x));
            total_layers[iLayer]->SetOutput(iNeuron, y);
            if (compute_gradient) {
              dy_dx = exp(-x) / pow(exp(-x) + 1, 2);
              for (auto iInput = 0u; iInput < inputLayer->GetNNeurons();
                  iInput++) {
                total_layers[iLayer]->SetdYdX(
                    iNeuron, iInput,
                    dy_dx * total_layers[iLayer]->GetdYdX(iNeuron, iInput));
              }
            }
          }
          break;
        case ENUM_ACTIVATION_FUNCTION::SELU:
          for (auto iNeuron = 0u; iNeuron < nNeurons_current; iNeuron++) {
            x = total_layers[iLayer]->GetInput(iNeuron);
            if (x > 0.0) {
              y = lambda * x;
              if (compute_gradient)
                dy_dx = lambda;
            } else {
              y = lambda * alpha * (exp(x) - 1);
              if (compute_gradient)
                dy_dx = lambda * alpha * exp(x);
            }
            total_layers[iLayer]->SetOutput(iNeuron, y);
            if (compute_gradient) {
              for (auto iInput = 0u; iInput < inputLayer->GetNNeurons();
                  iInput++) {
                total_layers[iLayer]->SetdYdX(
                    iNeuron, iInput,
                    dy_dx * total_layers[iLayer]->GetdYdX(iNeuron, iInput));
              }
            }
          }
          break;
        case ENUM_ACTIVATION_FUNCTION::GELU:
          for (auto iNeuron = 0u; iNeuron < nNeurons_current; iNeuron++) {
            x = total_layers[iLayer]->GetInput(iNeuron);

            y = 0.5 * x *
                (1 + tanh(0.7978845608028654 * (x + 0.044715 * pow(x, 3))));
            total_layers[iLayer]->SetOutput(iNeuron, y);
            if (compute_gradient) {
              dy_dx = 0.5 *
                      (tanh(0.0356774 * pow(x, 3) + 0.797885 * x) +
                      (0.107032 * pow(x, 3) + 0.797885 * x) * pow(cosh(x), -2) *
                          (0.0356774 * pow(x, 3) + 0.797885 * x));
              for (auto iInput = 0u; iInput < inputLayer->GetNNeurons();
                  iInput++) {
                total_layers[iLayer]->SetdYdX(
                    iNeuron, iInput,
                    dy_dx * total_layers[iLayer]->GetdYdX(iNeuron, iInput));
              }
            }
          }
          break;
        case ENUM_ACTIVATION_FUNCTION::NONE:
          for (auto iNeuron = 0u; iNeuron < nNeurons_current; iNeuron++) {
            y = 0.0;
            total_layers[iLayer]->SetOutput(iNeuron, y);
            if (compute_gradient) {
              dy_dx = 0.0;
              for (auto iInput = 0u; iInput < inputLayer->GetNNeurons();
                  iInput++) {
                total_layers[iLayer]->SetdYdX(
                    iNeuron, iInput,
                    dy_dx * total_layers[iLayer]->GetdYdX(iNeuron, iInput));
              }
            }
          }
          break;
        default:
          break;
        } // activation_function_types
      }
    }
    /* Compute and de-normalize MLP output */
    for (auto iNeuron = 0u; iNeuron < outputLayer->GetNNeurons(); iNeuron++) {
      mlpdouble y_norm = outputLayer->GetOutput(iNeuron);
      y = y_norm * (output_norm[iNeuron].second - output_norm[iNeuron].first) +
          output_norm[iNeuron].first;
      if (compute_gradient) {
        dy_dx = (output_norm[iNeuron].second - output_norm[iNeuron].first);
        for (auto iInput = 0u; iInput < inputLayer->GetNNeurons(); iInput++) {
          outputLayer->SetdYdX(iNeuron, iInput,
                              dy_dx * outputLayer->GetdYdX(iNeuron, iInput));
          dOutputs_dInputs[iNeuron][iInput] =
              outputLayer->GetdYdX(iNeuron, iInput);
        }
      }
      /* Storing output value */
      ANN_outputs[iNeuron] = y;
    }
  }

  /*!
   * \brief Set the normalization factors for the input layer
   * \param[in] iInput - Input index.
   * \param[in] input_min - Minimum input value.
   * \param[in] input_max - Maximum input value.
   */
  void SetInputNorm(unsigned long iInput, mlpdouble input_min, mlpdouble input_max) {
    input_norm[iInput] = std::make_pair(input_min, input_max);
  }

  /*!
   * \brief Set the normalization factors for the output layer
   * \param[in] iOutput - Input index.
   * \param[in] input_min - Minimum output value.
   * \param[in] input_max - Maximum output value.
   */
  void SetOutputNorm(unsigned long iOutput, mlpdouble output_min,
                     mlpdouble output_max) {
    output_norm[iOutput] = std::make_pair(output_min, output_max);
  }

  std::pair<mlpdouble, mlpdouble> GetInputNorm(unsigned long iInput) const {
    return input_norm[iInput];
  }

  std::pair<mlpdouble, mlpdouble> GetOutputNorm(unsigned long iOutput) const {
    return output_norm[iOutput];
  }
  /*!
   * \brief Add an output variable name to the network.
   * \param[in] input - Input variable name.
   */
  void SetOutputName(size_t iOutput, std::string input) {
    output_names[iOutput] = input;
  }

  /*!
   * \brief Add an input variable name to the network.
   * \param[in] output - Output variable name.
   */
  void SetInputName(size_t iInput, std::string input) {
    input_names[iInput] = input;
  }

  /*!
   * \brief Get network input variable name.
   * \param[in] iInput - Input variable index.
   * \returns input variable name.
   */
  std::string GetInputName(std::size_t iInput) const {
    return input_names[iInput];
  }

  /*!
   * \brief Get network output variable name.
   * \param[in] iOutput - Output variable index.
   * \returns output variable name.
   */
  std::string GetOutputName(std::size_t iOutput) const {
    return output_names[iOutput];
  }

  /*!
   * \brief Get network number of inputs.
   * \returns Number of network inputs
   */
  std::size_t GetnInputs() const { return input_names.size(); }

  /*!
   * \brief Get network number of outputs.
   * \returns Number of network outputs
   */
  std::size_t GetnOutputs() const { return output_names.size(); }

  /*!
   * \brief Get network evaluation output.
   * \param[in] iOutput - output index.
   * \returns Prediction value.
   */
  mlpdouble GetANNOutput(std::size_t iOutput) const {
    return ANN_outputs[iOutput];
  }

  /*!
   * \brief Get network output derivative w.r.t specific input.
   * \param[in] iOutput - output variable index.
   * \param[in] iInput - input variable index.
   * \returns Output derivative w.r.t input.
   */
  mlpdouble GetdOutputdInput(std::size_t iOutput, std::size_t iInput) const {
    return dOutputs_dInputs[iOutput][iInput];
  }

  /*!
   * \brief Set the activation function array size.
   * \param[in] n_layers - network layer count.
   */
  void SizeActivationFunctions(unsigned long n_layers) {
    activation_function_types.resize(n_layers);
    activation_function_names.resize(n_layers);
  }

  /*!
   * \brief Compute neuron activation function input.
   * \param[in] iLayer - Network layer index.
   * \param[in] iNeuron - Layer neuron index.
   * \returns Neuron activation function input.
   */
  mlpdouble ComputeX(std::size_t iLayer, std::size_t iNeuron) const {
    mlpdouble x;
    x = total_layers[iLayer]->GetBias(iNeuron);
    std::size_t nNeurons_previous = total_layers[iLayer - 1]->GetNNeurons();
    for (std::size_t jNeuron = 0; jNeuron < nNeurons_previous; jNeuron++) {
      x += weights_mat[iLayer - 1][iNeuron][jNeuron] *
           total_layers[iLayer - 1]->GetOutput(jNeuron);
    }
    return x;
  }
  mlpdouble ComputedOutputdInput(std::size_t iLayer, std::size_t iNeuron,
                              std::size_t iInput) const {
    mlpdouble doutput_dinput = 0;
    for (auto jNeuron = 0u; jNeuron < total_layers[iLayer - 1]->GetNNeurons();
         jNeuron++) {
      doutput_dinput += weights_mat[iLayer - 1][iNeuron][jNeuron] *
                        total_layers[iLayer - 1]->GetdYdX(jNeuron, iInput);
    }
    return doutput_dinput;
  }
};

} // namespace MLPToolbox
