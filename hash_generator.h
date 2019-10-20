#pragma once

#include <iostream>
#include <string>
#include <stdint.h>

#include <helper_functions.h> // includes for helper utility functions

typedef unsigned long long THash;

namespace hash_generator
{
	//void make_hash_from_key(const std::string &key_in, THash *hash)
	void make_hash_from_key(const unsigned char key_in[], THash *hash)
	{

		//uint8_t shift_offset = 7;
		uint8_t hash_iter = 0;
		unsigned int key_size = sizeof(&key_in) / sizeof(unsigned char);
		for (unsigned int ii = 0; ii < key_size; ii += 8)
		{
				THash tmp = (
			(THash)key_in[ii] << 56 |
			(THash)key_in[ii + 1] << 48 |
			(THash)key_in[ii + 2] << 40 |
			(THash)key_in[ii + 3] << 32 |
			(THash)key_in[ii + 4] << 24 |
			(THash)key_in[ii + 5] << 16 |
			(THash)key_in[ii + 6] << 8 |
			(THash)key_in[ii + 7] << 0);

			hash[hash_iter] = tmp;
			std::cout << std::hex << hash[hash_iter] << std::endl;

			hash_iter++;
			//shift_offset = 7;

		}

	};
}
