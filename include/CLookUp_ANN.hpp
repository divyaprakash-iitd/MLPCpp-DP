/*!
 * \file CLookUp_ANN.hpp
 * \brief Declaration of artificial neural network interpolation class
 * \author E.CBunschoten
 * \version 1.0.0
 */

#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "CIOMap.hpp"
#include "CNeuralNetwork.hpp"

namespace MLPToolbox {

class CLookUp_ANN {
  /*!
   *\class CLookUp_ANN
   *\brief This class allows for the evaluation of one or more multi-layer
   *perceptrons in for example thermodynamic state look-up operations. The
   *multi-layer perceptrons are loaded in the order listed in the MLP collection
   *file. Each multi-layer perceptron is generated based on the architecture
   *described in its respective input file. When evaluating the MLP collection,
   *an input-output map is used to find the correct MLP corresponding to the
   *call function inputs and outputs.
   */

private:
  std::vector<CNeuralNetwork> NeuralNetworks; /*!< std::std::vector containing
                                                 all loaded neural networks. */

  unsigned short number_of_variables; /*!< Number of loaded ANNs. */

  /*!
   * \brief Load ANN architecture
   * \param[in] ANN - pointer to target NeuralNetwork class
   * \param[in] filename - filename containing ANN architecture information
   */
  void GenerateANN(CNeuralNetwork &ANN, std::string filename);

public:
  /*!
   * \brief ANN collection class constructor
   * \param[in] n_inputs - Number of MLP files to be loaded.
   * \param[in] input_filenames - String array containing MLP input file names.
   */
  CLookUp_ANN(const unsigned short n_inputs,
              const std::string *input_filenames);

  /*!
   * \brief Evaluate loaded ANNs for given inputs and outputs
   * \param[in] input_output_map - input-output map coupling desired inputs and
   * outputs to loaded ANNs \param[in] inputs - input values \param[in] outputs
   * - pointers to output variables \returns Within output normalization range.
   */
  unsigned long PredictANN(CIOMap *input_output_map,
                           std::vector<double> &inputs,
                           std::vector<double *> &outputs);

  /*!
   * \brief Get number of loaded ANNs
   * \return number of loaded ANNs
   */
  std::size_t GetNANNs() const { return NeuralNetworks.size(); }

  /*!
   * \brief Check if all output variables are present in the loaded ANNs
   * \param[in] output_names - output variable names to check
   * \param[in] input_output_map - pointer to input-output map to be checked
   */
  bool CheckUseOfOutputs(std::vector<std::string> &output_names,
                         CIOMap *input_output_map) const;

  /*!
   * \brief Check if all input variables are present in the loaded ANNs
   * \param[in] input_names - input variable names to check
   * \param[in] input_output_map - pointer to input-output map to be checked
   */
  bool CheckUseOfInputs(std::vector<std::string> &input_names,
                        CIOMap *input_output_map) const;

  /*!
   * \brief Map variable names to ANN inputs or outputs
   * \param[in] i_ANN - loaded ANN index
   * \param[in] variable_names - variable names to map to ANN inputs or outputs
   * \param[in] input - map to inputs (true) or outputs (false)
   */
  std::vector<std::pair<std::size_t, std::size_t>>
  FindVariableIndices(std::size_t i_ANN,
                      std::vector<std::string> variable_names,
                      bool input) const;

  /*!
   * \brief Display architectural information on the loaded MLPs
   */
  void DisplayNetworkInfo() const;
};

} // namespace MLPToolbox
