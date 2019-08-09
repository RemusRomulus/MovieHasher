#pragma once

#include <iostream>
#include <string>
#include <stdint.h>

#include <helper_functions.h> // includes for helper utility functions

namespace hash_generator
{
	void make_hash_from_key(const std::string &key_in)
	{
		std::string buffered_key = key_in;
		size_t size = key_in.size();
		for (size_t ii = 3; ii < buffered_key.size(); ii += 4)
		{
			buffered_key.insert(ii, ".");
		}


		std::string output = "";
		{
			uint64_t hash[12];


			//uint8_t hash_count = 0;
			uint8_t shift_offset = 7;
			uint8_t hash_iter = 0;
			for (unsigned int ii = 0; ii < buffered_key.size(); ii += 8)
			{
				uint64_t tmp = (
					(uint64_t)buffered_key[ii] << 56 |
					(uint64_t)buffered_key[ii + 1] << 48 |
					(uint64_t)buffered_key[ii + 2] << 40 |
					(uint64_t)buffered_key[ii + 3] << 32 |
					(uint64_t)buffered_key[ii + 4] << 24 |
					(uint64_t)buffered_key[ii + 5] << 16 |
					(uint64_t)buffered_key[ii + 6] << 8 |
					(uint64_t)buffered_key[ii + 7] << 0);

				hash[hash_iter] = tmp;
				std::cout << std::hex << hash[hash_iter] << std::endl;

				hash_iter++;
				shift_offset = 7;

			}


			for (size_t jj = 0; jj < 12; jj++)
			{
				uint64_t tmpmask = 0x0000000000000001;
				uint64_t current_hash = hash[jj];
				for (size_t ll = 0; ll < 64; ll++)
				{
					unsigned char tmp = 255;
					uint64_t masked_hash = tmpmask & current_hash;
					bool flip = masked_hash;

					tmp = (flip) ? 255 : 0;
					output.push_back(tmp);
					tmpmask = tmpmask << 1;
				}

			}
			memset(hash, 0, 12 * sizeof(*hash));
		}

		
		sdkSavePPM4ub(std::string("NewHash.ppm").c_str(), (unsigned char*)output.c_str(), 16, 16);
		output.clear();
	};
}
