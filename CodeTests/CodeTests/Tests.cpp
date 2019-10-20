////////////////////////
// INCLUDES

//// System
#include <iostream>
#include <string>

//// User Defined
#include "../../Sequencer.h"
void createConstellation()
{

}

void ConstructCString()
{
	unsigned char glork[16] = {100, 12, 29, 255, 
								27, 128, 32, 109, 
								87, 91, 194, 5};

	char *glork2 = "abcdeiulahblivuahrlivghauier7yvao87foa87vff";

	std::string tmpstring(reinterpret_cast<char*> (glork));
	std::cout << tmpstring << "\n";
	std::cout << "Size of glork: " << sizeof(glork) << ". Size of &glork: " << sizeof(&glork) << "\n";
	std::cout << "Size of glork2: " << sizeof(glork2) << ". Size of &glork2: " << sizeof(&glork2) << "\n";
}


int main(int argc, char **argv)
{
	ConstructCString();
	
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