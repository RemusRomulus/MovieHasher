#pragma once
#include <regex>
#include <string>
#include <experimental/filesystem>


/** Sequencer
*/
/**
Code Copyright: Andrew Britton
Project: Movie Hasher
2019

Description:
Movie Integrity generator
*/


class Sequencer
{
public:

	//////////////////////
	// CTors / DTor
	Sequencer(std::string const first_frame_in): first_frame_name(first_frame_in) 
	{
		bool result = std::regex_search(first_frame_name, first_frame_regex_results, match_pattern);
		prefix = first_frame_regex_results[1];
		std::string frame = first_frame_regex_results[2];
		ext = first_frame_regex_results[3];
		first_frame = std::stoi(frame);
		digit_padding = frame.length();
		current_frame = first_frame;
		out_prefix = "_signature";
		file_delim = ".";

		get_last_frame();
	}

	~Sequencer() {}

	Sequencer(Sequencer const&) = delete;
	void operator=(Sequencer const&) = delete;


	bool get_next_frame(std::string &next_frame, std::string &name_out)
	{
		if (current_frame < last_frame)
		{
			tmp_num = add_padding(current_frame);
			next_frame = prefix + file_delim + tmp_num + file_delim + ext;
			name_out = prefix + out_prefix + file_delim + tmp_num + file_delim + "ppm";
			current_frame++;
			return true;
		}
		else
		{
			name_out = "";
			next_frame = "";
			return false;
		}
	}


private:

	void get_last_frame()
	{
		bool test = true;
		std::string tmpname;
		unsigned int i = current_frame + 1;
		do
		{
			tmpname = "C:\\ProgramData\\NVIDIA Corporation\\CUDA Samples\\v9.0\\3_Imaging\\MovieHasher\\data\\" + prefix + file_delim + add_padding(i) + file_delim + ext;
			test = std::experimental::filesystem::exists(tmpname);
			if (test)
				i++;
			else
				i--;
		} while (test);
		last_frame = i;
	}

	std::string add_padding(int frame_num_in)
	{
		std::string outval = std::to_string(frame_num_in);

		int length = outval.length();
		while (length < digit_padding)
		{
			outval = "0" + outval;
			length = outval.length();
		}

		return outval;
	}

	std::string first_frame_name;
	std::regex match_pattern = std::regex("([a-zA-Z]*).([0-9]*).(\\w{3})");
	std::smatch first_frame_regex_results;
	std::string prefix, ext, out_prefix, tmp_num, file_delim;
	int first_frame;
	int last_frame;
	int digit_padding;
	int current_frame;

};