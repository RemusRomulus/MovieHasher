#pragma once
#include <regex>
#include <string>
#include <experimental\filesystem>

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

		get_last_frame();
	}

	~Sequencer() {}

	Sequencer(Sequencer const&) = delete;
	void operator=(Sequencer const&) = delete;


	std::string get_next_frame();


private:

	void get_last_frame()
	{
		bool test = true;
		std::string tmpname;
		unsigned int i = current_frame + 1;
		do
		{
			tmpname = prefix + add_padding(i) + ext;
			test = std::experimental::filesystem::exists(tmpname);
		} while (test);

	}

	std::string add_padding(int frame_num_in)
	{
		std::string outval = std::to_string(frame_num_in);

		while (outval.length < digit_padding)
		{
			outval = "0" + outval;
		}

		return outval;
	}

	std::string first_frame_name;
	std::regex match_pattern = std::regex("([a-zA-Z]*).([0-9]*).(\\w{3})");
	std::smatch first_frame_regex_results;
	std::string prefix, ext;
	int first_frame;
	int last_frame;
	int digit_padding;
	int current_frame;

};