////////////////////////
// INCLUDES

//// System
#include <iostream>
#include <string>

//// User Defined
#include "../../Sequencer.h"

int main(int argc, char **argv)
{
	std::string image_filename = "prefix.00000.ext";
	std::string out_filename = "prefix_signature.00000.ppm";
	Sequencer sequencer(image_filename);

	bool tmp = true;
	do
	{
		tmp = sequencer.get_next_frame(image_filename, out_filename);
		if (tmp)
			std::cout << image_filename << ", " << out_filename << std::endl;
	} while (tmp);

	return 0;
}