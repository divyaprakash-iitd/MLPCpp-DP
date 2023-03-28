/*!
 * \file CNeuron.hpp
 * \brief Declaration of artificial neural network perceptron class
 * \author E.C.Bunschoten
 * \version 1.0.0
 */
#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#if defined(HAVE_OMP)
using su2double = codi::RealReverseIndexOpenMP;
#else
#if defined(CODI_INDEX_TAPE)
using su2double = codi::RealReverseIndex;
#else
using su2double = codi::RealReverse;
#endif
#endif
#elif defined(CODI_FORWARD_TYPE)  // forward mode AD
#include "codi.hpp"
using su2double = codi::RealForward;

#else  // primal / direct / no AD
using su2double = double;
#endif


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
  su2double output{0},       /*!< Output value of the current neuron */
      input{0},           /*!< Input value of the current neuron */
      doutput_dinput{0},  /*!< Gradient of output with respect to input */
      bias{0};            /*!< Bias value at current neuron */
  std::vector<su2double> doutput_dinputs;

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
  void SetOutput(su2double input) { output = input; }

  /*!
   * \brief Get neuron output value
   * \return Output value
   */
  su2double GetOutput() const { return output; }

  /*!
   * \brief Set neuron input value
   * \param[in] input - activation function input value
   */
  void SetInput(su2double x) { input = x; }

  /*!
   * \brief Get neuron input value
   * \return input value
   */
  su2double GetInput() const { return input; }

  /*!
   * \brief Set neuron bias
   * \param[in] input - bias value
   */
  void SetBias(su2double input) { bias = input; }

  /*!
   * \brief Get neuron bias value
   * \return bias value
   */
  su2double GetBias() const { return bias; }

  /*!
   * \brief Size the derivative of the neuron output wrt MLP inputs.
   * \param[in] nInputs - Number of MLP inputs.
   */
  void SizeGradient(std::size_t nInputs) { doutput_dinputs.resize(nInputs); }

  /*!
   * \brief Set neuron output gradient with respect to its input value
   * \param[in] input - Derivative of activation function with respect to input
   */
  void SetGradient(std::size_t iInput, su2double input) {
    doutput_dinputs[iInput] = input;
  }

  /*!
   * \brief Get neuron output gradient with respect to input value
   * \return output gradient wrt input value
   */
  su2double GetGradient(std::size_t iInput) const {
    return doutput_dinputs[iInput];
  }
};

} // namespace MLPToolbox
