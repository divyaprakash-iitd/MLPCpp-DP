/*!
 * \file CNeuron.hpp
 * \brief Declaration of artificial neural network perceptron class
 * \author E.C.Bunschoten
 * \version 1.1.0
 */
#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
#include "variable_def.hpp"

namespace MLPToolbox {
class CNeuron {
  /*!
   *\class CNeuron
   *\brief This class functions as a neuron within the CLayer class, making up
   *the CNeuralNetwork class. The CNeuron class functions as a location to store
   *activation function inputs and outputs, as well as gradients and biases.
   *These are accessed through the CLayer class for network evalution
   *operations.
   */
private:
  unsigned long i_neuron; /*!< Neuron identification number */
  mlpdouble output{0},       /*!< Output value of the current neuron */
      input{0},           /*!< Input value of the current neuron */
      doutput_dinput{0},  /*!< Gradient of output with respect to input */
      bias{0};            /*!< Bias value at current neuron */
  std::vector<mlpdouble> doutput_dinputs;

public:
  /*!
   * \brief Set neuron identification number
   * \param[in] input - Identification number
   */
  void SetNumber(unsigned long input) { i_neuron = input; }

  /*!
   * \brief Get neuron identification number
   * \return Identification number
   */
  unsigned long GetNumber() const { return i_neuron; }

  /*!
   * \brief Set neuron output value
   * \param[in] input - activation function output value
   */
  void SetOutput(mlpdouble input) { output = input; }

  /*!
   * \brief Get neuron output value
   * \return Output value
   */
  mlpdouble GetOutput() const { return output; }

  /*!
   * \brief Set neuron input value
   * \param[in] input - activation function input value
   */
  void SetInput(mlpdouble x) { input = x; }

  /*!
   * \brief Get neuron input value
   * \return input value
   */
  mlpdouble GetInput() const { return input; }

  /*!
   * \brief Set neuron bias
   * \param[in] input - bias value
   */
  void SetBias(mlpdouble input) { bias = input; }

  /*!
   * \brief Get neuron bias value
   * \return bias value
   */
  mlpdouble GetBias() const { return bias; }

  /*!
   * \brief Size the derivative of the neuron output wrt MLP inputs.
   * \param[in] nInputs - Number of MLP inputs.
   */
  void SizeGradient(std::size_t nInputs) { doutput_dinputs.resize(nInputs); }

  /*!
   * \brief Set neuron output gradient with respect to its input value
   * \param[in] input - Derivative of activation function with respect to input
   */
  void SetGradient(std::size_t iInput, mlpdouble input) {
    doutput_dinputs[iInput] = input;
  }

  /*!
   * \brief Get neuron output gradient with respect to input value
   * \return output gradient wrt input value
   */
  mlpdouble GetGradient(std::size_t iInput) const {
    return doutput_dinputs[iInput];
  }
};

} // namespace MLPToolbox
