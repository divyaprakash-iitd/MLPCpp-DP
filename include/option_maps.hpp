#include <string>
#include <map>

/*!
* \brief Available activation function enumeration.
*/
enum class ENUM_SCALING_FUNCTIONS {
MINMAX = 0,
STANDARD = 1,
ROBUST = 2,
};

/*!
* \brief Available activation function map.
*/
std::map<std::string, ENUM_SCALING_FUNCTIONS> scaling_map{
    {"minmax", ENUM_SCALING_FUNCTIONS::MINMAX},
    {"standard", ENUM_SCALING_FUNCTIONS::STANDARD},
    {"robust", ENUM_SCALING_FUNCTIONS::ROBUST},
};